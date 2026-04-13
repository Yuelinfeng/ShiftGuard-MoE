from __future__ import annotations

import hashlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
except Exception:
    DeepseekV2Config = None

try:
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Moe
except Exception:
    DeepseekV2Moe = None

from .workload import PromptSpec


@dataclass(slots=True)
class PromptTrace:
    prompt_id: str
    domain: str
    text: str
    token_count: int
    layer_accesses: dict[str, list[int]]
    generated_text: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DeepSeekRouteRecorder:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.layer_names: dict[int, str] = {}
        self._original_methods: list[tuple[DeepseekV2Moe, object]] = []
        self._current_prompt_id: str | None = None
        self._current_accesses: dict[int, list[int]] = defaultdict(list)

    def install(self) -> None:
        moe_layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if _is_deepseek_moe_module(module)
        ]
        if not moe_layers:
            raise RuntimeError("no DeepseekV2Moe layers found in model")

        for layer_idx, (layer_name, module) in enumerate(moe_layers):
            original = module.route_tokens_to_experts
            self.layer_names[layer_idx] = layer_name

            def wrapped_route(
                module_self,
                router_logits: torch.Tensor,
                *,
                _collector: DeepSeekRouteRecorder = self,
                _layer_idx: int = layer_idx,
                _original=original,
            ):
                topk_indices, topk_weights = _original(router_logits)
                if _collector._current_prompt_id is not None:
                    _collector._current_accesses[_layer_idx].extend(
                        topk_indices.reshape(-1).detach().cpu().tolist()
                    )
                return topk_indices, topk_weights

            module.route_tokens_to_experts = MethodType(wrapped_route, module)
            self._original_methods.append((module, original))

    def remove(self) -> None:
        for module, original in self._original_methods:
            module.route_tokens_to_experts = original
        self._original_methods.clear()

    @contextmanager
    def record(self, prompt_id: str) -> Iterator[None]:
        self._current_prompt_id = prompt_id
        self._current_accesses = defaultdict(list)
        try:
            yield
        finally:
            self._current_prompt_id = None

    def export_current_accesses(self) -> dict[str, list[int]]:
        return {
            str(layer_idx): list(self._current_accesses.get(layer_idx, []))
            for layer_idx in sorted(self.layer_names)
        }


class RealDeepSeekRunner:
    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.max_input_tokens = max_input_tokens
        self.tokenizer = None
        self.model = None
        self.recorder = None
        self.primary_device = torch.device("cpu")

    def load(self) -> None:
        model_kwargs: dict[str, object] = {
            "low_cpu_mem_usage": True,
        }
        resolved_dtype = _resolve_dtype(self.dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        if self.device_map == "auto":
            model_kwargs["device_map"] = "auto"

        last_error: Exception | None = None

        if DeepseekV2Moe is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=False,
                    **model_kwargs,
                )
            except Exception as exc:
                last_error = exc
                self.tokenizer = None
                self.model = None

        if self.model is None or self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    **model_kwargs,
                )
            except Exception as exc:
                if "flash_attn" in str(exc):
                    raise RuntimeError(
                        "Falling back to trust_remote_code=True requires flash_attn in this environment. "
                        "Prefer upgrading transformers to a version with native deepseek_v2 support, "
                        "or install flash_attn first."
                    ) from exc
                if last_error is not None:
                    raise RuntimeError(
                        "Failed to load DeepSeek-V2 using both native transformers support and "
                        "trust_remote_code fallback."
                    ) from exc
                raise

        if self.device_map not in {"auto", ""}:
            self.model = self.model.to(self.device_map)
        self.model.eval()
        self.primary_device = next(self.model.parameters()).device
        self.recorder = DeepSeekRouteRecorder(self.model)
        self.recorder.install()

    def trace_prompt(self, prompt: PromptSpec, max_new_tokens: int = 0) -> PromptTrace:
        if self.model is None or self.tokenizer is None or self.recorder is None:
            raise RuntimeError("runner is not loaded")

        encoded = self._encode_chat_prompt(prompt.text)
        encoded = {key: value.to(self.primary_device) for key, value in encoded.items()}
        generated_text = None

        with torch.inference_mode():
            with self.recorder.record(prompt.prompt_id):
                if max_new_tokens > 0:
                    pad_token_id = (
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.model.config.eos_token_id
                    )
                    output_ids = self.model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=pad_token_id,
                    )
                    new_tokens = output_ids[0, encoded["input_ids"].shape[-1] :]
                    generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                else:
                    self.model(**encoded, use_cache=False)

        return PromptTrace(
            prompt_id=prompt.prompt_id,
            domain=prompt.domain,
            text=prompt.text,
            token_count=int(encoded["input_ids"].shape[-1]),
            layer_accesses=self.recorder.export_current_accesses(),
            generated_text=generated_text,
        )

    def _encode_chat_prompt(self, text: str) -> dict[str, torch.Tensor]:
        messages = [{"role": "user", "content": text}]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            encoded = {"input_ids": input_ids}
        except Exception:
            encoded = self.tokenizer(text, return_tensors="pt")

        input_ids = encoded["input_ids"][:, -self.max_input_tokens :]
        attention_mask = encoded.get("attention_mask")
        output = {"input_ids": input_ids}
        if attention_mask is not None:
            output["attention_mask"] = attention_mask[:, -self.max_input_tokens :]
        return output


class TinyRandomDeepSeekRunner:
    def __init__(self, max_input_tokens: int = 128) -> None:
        self.max_input_tokens = max_input_tokens
        self.model = None
        self.recorder = None
        self.vocab_size = 256

    def load(self) -> None:
        if DeepseekV2Config is None:
            raise RuntimeError(
                "tiny-random mode requires a transformers build that includes "
                "transformers.models.deepseek_v2. Upgrade transformers or use --mode real."
            )
        config = DeepseekV2Config(
            vocab_size=self.vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            max_position_embeddings=max(256, self.max_input_tokens + 16),
        )
        self.model = AutoModelForCausalLM.from_config(config).eval()
        self.recorder = DeepSeekRouteRecorder(self.model)
        self.recorder.install()

    def trace_prompt(self, prompt: PromptSpec, max_new_tokens: int = 0) -> PromptTrace:
        if self.model is None or self.recorder is None:
            raise RuntimeError("runner is not loaded")

        input_ids = self._hash_encode(prompt.text)
        with torch.inference_mode():
            with self.recorder.record(prompt.prompt_id):
                self.model(input_ids=input_ids, use_cache=False)

        return PromptTrace(
            prompt_id=prompt.prompt_id,
            domain=prompt.domain,
            text=prompt.text,
            token_count=int(input_ids.shape[-1]),
            layer_accesses=self.recorder.export_current_accesses(),
            generated_text=None,
        )

    def _hash_encode(self, text: str) -> torch.Tensor:
        tokens = [1]
        for piece in text.replace("\n", " ").split():
            digest = hashlib.sha1(piece.encode("utf-8")).hexdigest()
            token = 3 + (int(digest[:8], 16) % (self.vocab_size - 3))
            tokens.append(token)
            if len(tokens) >= self.max_input_tokens - 1:
                break
        tokens.append(2)
        return torch.tensor([tokens], dtype=torch.long)


def _resolve_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name in {"none", ""}:
        return None
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _is_deepseek_moe_module(module: torch.nn.Module) -> bool:
    if DeepseekV2Moe is not None and isinstance(module, DeepseekV2Moe):
        return True

    class_name = module.__class__.__name__.lower()
    return "deepseek" in class_name and "moe" in class_name and hasattr(module, "route_tokens_to_experts")
