from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ShiftGuard-MoE experiment outputs and generate plots."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory that contains summary.json, prompt_metrics.csv, and workloads.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for analysis outputs. Defaults to <input-dir>/analysis.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=6,
        help="Rolling window size for prompt-level hit-rate curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(input_dir / "summary.json")
    prompt_metrics = _load_prompt_metrics(input_dir / "prompt_metrics.csv")

    scenario_rows = _build_scenario_rows(summary)
    _write_csv(output_dir / "scenario_summary.csv", scenario_rows)
    _plot_total_hit_rate(summary, output_dir / "scenario_total_hit_rate.png")
    _plot_drop_vs_stable(summary, output_dir / "scenario_drop_vs_stable.png")
    _plot_window_trends(summary, output_dir / "window_trends_by_policy.png")
    _plot_prompt_curves(
        prompt_metrics=prompt_metrics,
        output_path=output_dir / "prompt_rolling_hit_rate.png",
        rolling_window=args.rolling_window,
    )
    _write_markdown_report(
        summary=summary,
        scenario_rows=scenario_rows,
        output_path=output_dir / "analysis_report.md",
    )

    print(f"Analysis finished. Files saved to: {output_dir}")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prompt_metrics(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "scenario": row["scenario"],
                    "policy": row["policy"],
                    "position": int(row["position"]),
                    "window_index": int(row["window_index"]),
                    "prompt_id": row["prompt_id"],
                    "domain": row["domain"],
                    "hits": int(row["hits"]),
                    "misses": int(row["misses"]),
                    "hit_rate": float(row["hit_rate"]),
                    "cumulative_hit_rate": float(row["cumulative_hit_rate"]),
                }
            )
    return rows


def _build_scenario_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scenarios = summary["scenarios"]
    for scenario_name, policies in scenarios.items():
        for policy_name, metrics in policies.items():
            rows.append(
                {
                    "scenario": scenario_name,
                    "policy": policy_name,
                    "total_hit_rate": metrics["total_hit_rate"],
                    "first_window_hit_rate": metrics["first_window_hit_rate"],
                    "post_shift_hit_rate": metrics["post_shift_hit_rate"],
                    "worst_window_hit_rate": metrics["worst_window_hit_rate"],
                    "drop_vs_stable": metrics["drop_vs_stable"],
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_total_hit_rate(summary: dict[str, object], output_path: Path) -> None:
    scenarios = list(summary["scenarios"].keys())
    policies = _policy_names(summary)
    x_positions = list(range(len(scenarios)))
    width = 0.35 if len(policies) == 2 else 0.8 / max(1, len(policies))

    fig, axis = plt.subplots(figsize=(10, 4.8))
    for policy_index, policy_name in enumerate(policies):
        values = [
            float(summary["scenarios"][scenario][policy_name]["total_hit_rate"])
            for scenario in scenarios
        ]
        offsets = [x + (policy_index - (len(policies) - 1) / 2) * width for x in x_positions]
        axis.bar(offsets, values, width=width, label=policy_name.upper())

    axis.set_xticks(x_positions)
    axis.set_xticklabels(scenarios)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Total hit rate")
    axis.set_title("Policy hit rate by scenario")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_drop_vs_stable(summary: dict[str, object], output_path: Path) -> None:
    scenarios = list(summary["scenarios"].keys())
    policies = _policy_names(summary)
    x_positions = list(range(len(scenarios)))
    width = 0.35 if len(policies) == 2 else 0.8 / max(1, len(policies))

    fig, axis = plt.subplots(figsize=(10, 4.8))
    for policy_index, policy_name in enumerate(policies):
        values = [
            float(summary["scenarios"][scenario][policy_name]["drop_vs_stable"])
            for scenario in scenarios
        ]
        offsets = [x + (policy_index - (len(policies) - 1) / 2) * width for x in x_positions]
        axis.bar(offsets, values, width=width, label=policy_name.upper())

    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(scenarios)
    axis.set_ylabel("Drop vs stable")
    axis.set_title("How much each scenario degrades from stable")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_window_trends(summary: dict[str, object], output_path: Path) -> None:
    policies = _policy_names(summary)
    fig, axes = plt.subplots(len(policies), 1, figsize=(10, 4 * len(policies)), sharex=True)
    if len(policies) == 1:
        axes = [axes]

    for axis, policy_name in zip(axes, policies):
        for scenario_name, scenario_metrics in summary["scenarios"].items():
            windows = scenario_metrics[policy_name]["window_hit_rates"]
            x_values = [int(item["window_index"]) for item in windows]
            y_values = [float(item["average_hit_rate"]) for item in windows]
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=scenario_name)

        axis.set_title(f"{policy_name.upper()} window trend")
        axis.set_ylabel("Average window hit rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Window index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_prompt_curves(
    *,
    prompt_metrics: list[dict[str, object]],
    output_path: Path,
    rolling_window: int,
) -> None:
    scenarios = sorted({str(row["scenario"]) for row in prompt_metrics})
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 3.2 * len(scenarios)), sharex=False)
    if len(scenarios) == 1:
        axes = [axes]

    for axis, scenario in zip(axes, scenarios):
        rows = [row for row in prompt_metrics if row["scenario"] == scenario]
        policies = sorted({str(row["policy"]) for row in rows})
        for policy_name in policies:
            policy_rows = [row for row in rows if row["policy"] == policy_name]
            policy_rows.sort(key=lambda item: int(item["position"]))
            x_values = [int(row["position"]) for row in policy_rows]
            y_values = [float(row["hit_rate"]) for row in policy_rows]
            axis.plot(
                x_values,
                _rolling_average(y_values, rolling_window),
                linewidth=2,
                label=policy_name.upper(),
            )

        axis.set_title(f"{scenario} prompt-level rolling hit rate")
        axis.set_ylabel("Rolling hit rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Prompt position")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_markdown_report(
    *,
    summary: dict[str, object],
    scenario_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    stable_rows = [row for row in scenario_rows if row["scenario"] == "stable"]
    non_stable_rows = [row for row in scenario_rows if row["scenario"] != "stable"]

    biggest_drop = max(non_stable_rows, key=lambda row: float(row["drop_vs_stable"])) if non_stable_rows else None
    best_non_stable = min(non_stable_rows, key=lambda row: float(row["drop_vs_stable"])) if non_stable_rows else None

    lines: list[str] = []
    lines.append("# Analysis Report")
    lines.append("")
    lines.append("## Experiment")
    lines.append("")
    lines.append(f"- Mode: `{summary['mode']}`")
    lines.append(f"- Model: `{summary['model_id']}`")
    lines.append(f"- Cache capacity: `{summary['cache_capacity']}`")
    lines.append(f"- Number of prompts: `{summary['num_unique_prompts']}`")
    lines.append(f"- Number of MoE layers traced: `{summary['num_moe_layers']}`")
    lines.append("")
    lines.append("## Stable Baseline")
    lines.append("")
    for row in stable_rows:
        lines.append(
            f"- `{row['policy'].upper()}` stable total hit rate: "
            f"`{float(row['total_hit_rate']):.4f}`"
        )
    lines.append("")
    lines.append("## Scenario Comparison")
    lines.append("")
    lines.append("| Scenario | Policy | Total Hit Rate | Post-shift Hit Rate | Drop vs Stable | Worst Window |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in scenario_rows:
        lines.append(
            f"| {row['scenario']} | {str(row['policy']).upper()} | "
            f"{float(row['total_hit_rate']):.4f} | "
            f"{float(row['post_shift_hit_rate']):.4f} | "
            f"{float(row['drop_vs_stable']):.4f} | "
            f"{float(row['worst_window_hit_rate']):.4f} |"
        )
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    if biggest_drop is not None:
        lines.append(
            f"- Largest degradation: `{biggest_drop['scenario']}` with `{str(biggest_drop['policy']).upper()}` "
            f"showing `drop_vs_stable={float(biggest_drop['drop_vs_stable']):.4f}`."
        )
    if best_non_stable is not None:
        lines.append(
            f"- Most resilient non-stable case: `{best_non_stable['scenario']}` with "
            f"`{str(best_non_stable['policy']).upper()}` at "
            f"`drop_vs_stable={float(best_non_stable['drop_vs_stable']):.4f}`."
        )

    per_scenario = defaultdict(list)
    for row in non_stable_rows:
        per_scenario[str(row["scenario"])].append(row)
    for scenario_name, rows in per_scenario.items():
        lru_row = next((row for row in rows if row["policy"] == "lru"), None)
        lfu_row = next((row for row in rows if row["policy"] == "lfu"), None)
        if lru_row and lfu_row:
            lines.append(
                f"- `{scenario_name}`: `LRU drop={float(lru_row['drop_vs_stable']):.4f}`, "
                f"`LFU drop={float(lfu_row['drop_vs_stable']):.4f}`."
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rolling_average(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        bucket = values[start : idx + 1]
        output.append(sum(bucket) / len(bucket))
    return output


def _policy_names(summary: dict[str, object]) -> list[str]:
    first_scenario = next(iter(summary["scenarios"].values()))
    return list(first_scenario.keys())


if __name__ == "__main__":
    main()
