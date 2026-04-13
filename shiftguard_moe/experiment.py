from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from .policies import LFUExpertCache, LRUExpertCache, LayeredCacheSimulator
from .runner import PromptTrace, RealDeepSeekRunner, TinyRandomDeepSeekRunner
from .workload import PromptSpec, WorkloadItem, build_workloads


def run_experiment(
    *,
    output_dir: str | Path,
    prompt_bank: list[PromptSpec],
    workload_map: dict[str, list[WorkloadItem]] | None = None,
    cache_capacity: int = 8,
    mode: str = "real",
    model_id: str = "deepseek-ai/DeepSeek-V2-Lite-Chat",
    device_map: str = "auto",
    dtype: str = "auto",
    max_input_tokens: int = 512,
    max_new_tokens: int = 0,
    traces: dict[str, PromptTrace] | None = None,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    workload_map = workload_map or build_workloads(prompt_bank)

    if traces is None:
        runner = _build_runner(
            mode=mode,
            model_id=model_id,
            device_map=device_map,
            dtype=dtype,
            max_input_tokens=max_input_tokens,
        )
        runner.load()
        traces = _trace_prompt_bank(runner, prompt_bank, max_new_tokens=max_new_tokens)

    save_traces_json(traces, output_path / "prompt_traces.json")
    _save_prompt_bank(prompt_bank, output_path / "prompt_bank.jsonl")
    _save_workloads_json(workload_map, output_path / "workloads.json")

    prompt_metrics: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "mode": mode,
        "model_id": model_id if mode == "real" else "tiny-random-deepseek-v2",
        "cache_capacity": cache_capacity,
        "num_unique_prompts": len(prompt_bank),
        "num_moe_layers": _infer_num_layers(traces),
        "domain_js_divergence": _domain_divergence(traces),
        "scenarios": {},
    }

    policy_specs = {
        "lru": LRUExpertCache,
        "lfu": LFUExpertCache,
    }
    num_layers = int(summary["num_moe_layers"])

    for scenario_name, workload in workload_map.items():
        summary["scenarios"][scenario_name] = {}
        for policy_name, policy_cls in policy_specs.items():
            simulator = LayeredCacheSimulator(policy_cls=policy_cls, num_layers=num_layers, capacity=cache_capacity)
            rows: list[dict[str, object]] = []
            for item in workload:
                trace = traces[item.prompt_id]
                stats = simulator.consume(trace.layer_accesses)
                rows.append(
                    {
                        "scenario": scenario_name,
                        "policy": policy_name,
                        "position": item.position,
                        "window_index": item.window_index,
                        "prompt_id": item.prompt_id,
                        "domain": item.domain,
                        "hits": stats.hits,
                        "misses": stats.misses,
                        "hit_rate": stats.hit_rate,
                        "cumulative_hit_rate": simulator.cumulative_hit_rate,
                    }
                )
            prompt_metrics.extend(rows)
            summary["scenarios"][scenario_name][policy_name] = _summarize_rows(rows)

    _attach_drop_from_stable(summary)
    _save_prompt_metrics(prompt_metrics, output_path / "prompt_metrics.csv")
    _plot_prompt_hit_rates(prompt_metrics, output_path / "prompt_hit_rates.png")
    _plot_window_hit_rates(summary["scenarios"], output_path / "window_hit_rates.png")

    summary_path = output_path / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def save_traces_json(traces: dict[str, PromptTrace], path: str | Path) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump({key: trace.to_dict() for key, trace in traces.items()}, handle, ensure_ascii=False, indent=2)


def load_traces_json(path: str | Path) -> dict[str, PromptTrace]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    traces: dict[str, PromptTrace] = {}
    for prompt_id, item in data.items():
        traces[prompt_id] = PromptTrace(
            prompt_id=item["prompt_id"],
            domain=item["domain"],
            text=item["text"],
            token_count=item["token_count"],
            layer_accesses={str(key): value for key, value in item["layer_accesses"].items()},
            generated_text=item.get("generated_text"),
        )
    return traces


def _build_runner(
    *,
    mode: str,
    model_id: str,
    device_map: str,
    dtype: str,
    max_input_tokens: int,
):
    if mode == "real":
        return RealDeepSeekRunner(
            model_id=model_id,
            device_map=device_map,
            dtype=dtype,
            max_input_tokens=max_input_tokens,
        )
    if mode == "tiny-random":
        return TinyRandomDeepSeekRunner(max_input_tokens=max_input_tokens)
    raise ValueError(f"unsupported mode: {mode}")


def _trace_prompt_bank(runner, prompt_bank: list[PromptSpec], max_new_tokens: int = 0) -> dict[str, PromptTrace]:
    traces: dict[str, PromptTrace] = {}
    for prompt in prompt_bank:
        traces[prompt.prompt_id] = runner.trace_prompt(prompt, max_new_tokens=max_new_tokens)
    return traces


def _infer_num_layers(traces: dict[str, PromptTrace]) -> int:
    sample = next(iter(traces.values()))
    return len(sample.layer_accesses)


def _domain_divergence(traces: dict[str, PromptTrace]) -> dict[str, dict[str, float]]:
    per_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces.values():
        for layer_key, expert_ids in trace.layer_accesses.items():
            for expert_id in expert_ids:
                per_domain[trace.domain][f"L{layer_key}:E{expert_id}"] += 1

    domains = sorted(per_domain)
    matrix: dict[str, dict[str, float]] = {}
    for domain_a in domains:
        matrix[domain_a] = {}
        for domain_b in domains:
            matrix[domain_a][domain_b] = round(_js_divergence(per_domain[domain_a], per_domain[domain_b]), 6)
    return matrix


def _js_divergence(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    keys = set(counter_a) | set(counter_b)
    if not keys:
        return 0.0
    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())
    probs_a = {key: counter_a[key] / total_a for key in keys}
    probs_b = {key: counter_b[key] / total_b for key in keys}
    mixed = {key: 0.5 * (probs_a[key] + probs_b[key]) for key in keys}
    return 0.5 * (_kl_divergence(probs_a, mixed) + _kl_divergence(probs_b, mixed))


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    value = 0.0
    for key, p_value in p.items():
        if p_value == 0.0:
            continue
        q_value = q[key]
        value += p_value * math.log(p_value / q_value, 2)
    return value


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    total_hits = sum(int(row["hits"]) for row in rows)
    total_misses = sum(int(row["misses"]) for row in rows)
    total_accesses = total_hits + total_misses
    window_groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        window_groups[int(row["window_index"])].append(float(row["hit_rate"]))

    window_hit_rates = [
        {
            "window_index": window_index,
            "average_hit_rate": sum(values) / len(values),
        }
        for window_index, values in sorted(window_groups.items())
    ]

    first_window_hit_rate = window_hit_rates[0]["average_hit_rate"] if window_hit_rates else 0.0
    post_shift_values = [item["average_hit_rate"] for item in window_hit_rates[1:]]
    post_shift_hit_rate = sum(post_shift_values) / len(post_shift_values) if post_shift_values else first_window_hit_rate

    return {
        "total_accesses": total_accesses,
        "total_hit_rate": 0.0 if total_accesses == 0 else total_hits / total_accesses,
        "first_window_hit_rate": first_window_hit_rate,
        "post_shift_hit_rate": post_shift_hit_rate,
        "worst_window_hit_rate": min((item["average_hit_rate"] for item in window_hit_rates), default=0.0),
        "window_hit_rates": window_hit_rates,
    }


def _attach_drop_from_stable(summary: dict[str, object]) -> None:
    scenarios = summary["scenarios"]
    if "stable" not in scenarios:
        return
    stable = scenarios["stable"]
    for scenario_name, policies in scenarios.items():
        for policy_name, metrics in policies.items():
            stable_hit_rate = stable[policy_name]["total_hit_rate"]
            metrics["drop_vs_stable"] = stable_hit_rate - metrics["total_hit_rate"]


def _save_prompt_bank(prompt_bank: list[PromptSpec], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompt_bank:
            handle.write(json.dumps(asdict(prompt), ensure_ascii=False) + "\n")


def _save_workloads_json(workloads: dict[str, list[WorkloadItem]], path: Path) -> None:
    serializable = {
        scenario: [asdict(item) for item in items]
        for scenario, items in workloads.items()
    }
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_prompt_metrics(prompt_metrics: list[dict[str, object]], path: Path) -> None:
    if not prompt_metrics:
        return
    fieldnames = list(prompt_metrics[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prompt_metrics)


def _plot_prompt_hit_rates(prompt_metrics: list[dict[str, object]], path: Path) -> None:
    scenarios = sorted({str(row["scenario"]) for row in prompt_metrics})
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 3.2 * len(scenarios)), sharex=False)
    if len(scenarios) == 1:
        axes = [axes]

    for axis, scenario in zip(axes, scenarios):
        rows = [row for row in prompt_metrics if row["scenario"] == scenario]
        for policy in sorted({str(row["policy"]) for row in rows}):
            policy_rows = [row for row in rows if row["policy"] == policy]
            policy_rows.sort(key=lambda item: int(item["position"]))
            x_values = [int(row["position"]) for row in policy_rows]
            y_values = [float(row["hit_rate"]) for row in policy_rows]
            axis.plot(x_values, _rolling_average(y_values, window=4), label=policy.upper(), linewidth=2)

        axis.set_title(f"{scenario} workload")
        axis.set_ylabel("Rolling hit rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Prompt position")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_window_hit_rates(scenarios: dict[str, dict[str, object]], path: Path) -> None:
    scenario_names = list(scenarios)
    x_values = range(len(scenario_names))
    lru_values = [float(scenarios[scenario]["lru"]["post_shift_hit_rate"]) for scenario in scenario_names]
    lfu_values = [float(scenarios[scenario]["lfu"]["post_shift_hit_rate"]) for scenario in scenario_names]

    fig, axis = plt.subplots(figsize=(10, 4.5))
    width = 0.35
    axis.bar([x - width / 2 for x in x_values], lru_values, width=width, label="LRU")
    axis.bar([x + width / 2 for x in x_values], lfu_values, width=width, label="LFU")
    axis.set_xticks(list(x_values))
    axis.set_xticklabels(scenario_names)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Average post-shift hit rate")
    axis.set_title("Policy hit rate after workload shift")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _rolling_average(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        bucket = values[start : idx + 1]
        output.append(sum(bucket) / len(bucket))
    return output
