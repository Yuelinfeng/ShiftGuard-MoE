from __future__ import annotations

import argparse
import json
from pathlib import Path

from shiftguard_moe.experiment import load_traces_json, run_experiment
from shiftguard_moe.workload import (
    DEFAULT_SCENARIOS,
    available_domains,
    build_workloads,
    load_custom_prompt_bank,
    load_workload_plan,
    make_builtin_prompt_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal DeepSeek-V2-Lite-Chat MoE workload-shift experiment."
    )
    parser.add_argument("--mode", choices=["real", "tiny-random"], default="tiny-random")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--model-name-or-path", dest="model_name_or_path", default="")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--workload-plan-path", type=str, default="")
    parser.add_argument("--domains", type=str, default="code,math,medicine,creative")
    parser.add_argument("--prompts-per-domain", type=int, default=8)
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--num-windows", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--cache-capacity", type=int, default=8)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="outputs/latest")
    parser.add_argument("--load-traces-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_name_or_path:
        args.model_id = args.model_name_or_path
    scenarios = _split_csv(args.scenarios)
    domains = _split_csv(args.domains)

    if args.dataset_path:
        prompt_bank = load_custom_prompt_bank(args.dataset_path)
    else:
        if not domains:
            domains = available_domains()
        prompt_bank = make_builtin_prompt_bank(
            domains=domains,
            prompts_per_domain=args.prompts_per_domain,
            seed=args.seed,
        )

    if args.workload_plan_path:
        workloads = load_workload_plan(
            prompt_bank=prompt_bank,
            plan_path=args.workload_plan_path,
            seed=args.seed,
        )
    else:
        workloads = build_workloads(
            prompt_bank=prompt_bank,
            scenarios=scenarios,
            num_windows=args.num_windows,
            window_size=args.window_size,
            seed=args.seed,
        )

    traces = load_traces_json(args.load_traces_json) if args.load_traces_json else None
    summary = run_experiment(
        output_dir=args.output_dir,
        prompt_bank=prompt_bank,
        workload_map=workloads,
        cache_capacity=args.cache_capacity,
        mode=args.mode,
        model_id=args.model_id,
        device_map=args.device_map,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        traces=traces,
    )

    summary_path = Path(args.output_dir) / "summary.json"
    print(f"Experiment finished. Summary saved to: {summary_path}")
    print(json.dumps(summary["scenarios"], ensure_ascii=False, indent=2))


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
