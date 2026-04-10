"""
Run the main quantitative experiments across all supported models and write
text-first summaries instead of plots.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import compute_head_dla_batch, dataset_activation_patching_experiment, dataset_amplification_experiment, select_top_heads
from src.benchmark import analyse_sgr_distribution, run_benchmark
from src.dataset import load_counterfact
from src.models import CANONICAL_MODEL_NAMES, MODEL_SHORTNAMES, load_model
from src.utils import benchmark_csv_path


torch.manual_seed(67)


def _progress(step: int, total: int, message: str) -> None:
    print(f"[Step {step:02d}/{total:02d}] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-model rerun with text-only reporting")
    parser.add_argument("--models", nargs="+", default=CANONICAL_MODEL_NAMES, help="Models to benchmark")
    parser.add_argument("--negator_suffix", default=" not", help="Negator suffix used to build prompt pairs")
    parser.add_argument("--max_samples", type=int, default=200, help="CounterFact samples per model (-1 = all)")
    parser.add_argument("--analysis_samples", type=int, default=200, help="Samples used for head selection and amplification")
    parser.add_argument("--patching_samples", type=int, default=20, help="Samples used for dataset activation patching")
    parser.add_argument("--top_k_heads", type=int, default=10, help="Top inhibition heads to summarize")
    parser.add_argument("--amp_scales", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0], help="Amplification scales")
    parser.add_argument("--results_dir", default="results/cross_model", help="Output directory for CSVs")
    parser.add_argument("--report_md", default="reports/cross_model_experiments.md", help="Markdown report path")
    return parser.parse_args()


def _markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> list[str]:
    if df.empty:
        return ["(no data)", ""]

    float_cols = float_cols or set()
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join(["---"] * len(df.columns)) + "|",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if col in float_cols and pd.notna(value):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def _top_heads_string(top_heads: list[tuple[int, int]], limit: int = 5) -> str:
    preview = top_heads[:limit]
    suffix = ", ..." if len(top_heads) > limit else ""
    return ", ".join(f"({layer},{head})" for layer, head in preview) + suffix


def _supportive_findings(
    benchmark_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    amplification_df: pd.DataFrame,
    patching_df: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []

    for _, row in benchmark_df.iterrows():
        if pd.notna(row["failure_median_sgr"]) and pd.notna(row["success_median_sgr"]) and row["failure_median_sgr"] > row["success_median_sgr"]:
            findings.append(
                f"`{row['model_name']}` shows higher median SGR in failures than successes "
                f"({row['failure_median_sgr']:.2f} vs {row['success_median_sgr']:.2f}), which supports the retrieval-over-inhibition story."
            )

    for _, row in amplification_df.iterrows():
        if pd.notna(row["absolute_improvement"]) and row["absolute_improvement"] > 0:
            findings.append(
                f"`{row['model_name']}` improves under head amplification: failure rate drops from "
                f"{row['baseline_rate']:.1%} to {row['best_rate']:.1%} at scale {row['best_scale']:.2f}."
            )

    for _, row in patching_df.iterrows():
        if pd.notna(row["best_delta"]) and row["best_delta"] > 0:
            findings.append(
                f"`{row['model_name']}` has a positive mean patching effect, strongest for "
                f"{row['best_patch_type']} at layer {int(row['best_layer'])} (Δ logit {row['best_delta']:+.3f})."
            )

    for _, row in mismatch_df.iterrows():
        if pd.notna(row["failure_mismatch_rate"]) and row["failure_mismatch_rate"] < 0.1:
            findings.append(
                f"`{row['model_name']}` rarely produces failure cases with SGR <= 1 "
                f"({row['failure_mismatch_rate']:.1%}), so the metric usually points in the right direction for failures."
            )

    return findings


def _contradictory_findings(
    benchmark_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    amplification_df: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []

    for _, row in mismatch_df.iterrows():
        if pd.notna(row["success_mismatch_rate"]) and row["success_mismatch_rate"] > 0:
            findings.append(
                f"`{row['model_name']}` still has many successful suppressions with SGR > 1 "
                f"({row['success_mismatch_rate']:.1%} of successes), so SGR is a useful trend metric, not a clean decision rule."
            )
        if pd.notna(row["failure_mismatch_rate"]) and row["failure_mismatch_rate"] > 0:
            findings.append(
                f"`{row['model_name']}` also includes some failure cases with SGR <= 1 "
                f"({row['failure_mismatch_rate']:.1%} of failures), which weakens any strict threshold claim."
            )

    for _, row in amplification_df.iterrows():
        if pd.notna(row["best_rate"]) and row["best_rate"] > 0:
            findings.append(
                f"`{row['model_name']}` is not fully rescued by amplification; even the best scale leaves a "
                f"{row['best_rate']:.1%} failure rate."
            )

    failure_order = benchmark_df.sort_values("failure_rate", ascending=False)
    if len(failure_order) >= 2:
        worst = failure_order.iloc[0]
        best = failure_order.iloc[-1]
        findings.append(
            f"Model differences remain large: `{worst['model_name']}` fails more often than `{best['model_name']}` "
            f"({worst['failure_rate']:.1%} vs {best['failure_rate']:.1%}), so the story is not equally strong across architectures."
        )

    return findings


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    model_names = [MODEL_SHORTNAMES.get(model_name, model_name) for model_name in args.models]
    max_samples = None if args.max_samples < 0 else args.max_samples

    per_model_steps = 6
    finalization_steps = 10
    total_steps = len(model_names) * per_model_steps + finalization_steps
    current_step = 0

    all_dfs: list[pd.DataFrame] = []
    head_rows: list[dict[str, object]] = []
    amp_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []

    current_step += 1
    _progress(
        current_step,
        total_steps,
        f"Initialized run for {len(model_names)} model(s): {', '.join(model_names)}",
    )

    for model_name in model_names:
        current_step += 1
        _progress(current_step, total_steps, f"Loading model `{model_name}`")
        print("\n" + "#" * 72)
        print(f"# Running model: {model_name}")
        print("#" * 72 + "\n")

        model = load_model(model_name)

        current_step += 1
        _progress(current_step, total_steps, f"Building CounterFact pairs for `{model_name}`")
        pairs = load_counterfact(max_samples=max_samples, model=model, negator_suffix=args.negator_suffix)

        current_step += 1
        _progress(current_step, total_steps, f"Running benchmark + saving per-model CSV for `{model_name}`")
        csv_path = str(benchmark_csv_path(args.results_dir, model_name, args.negator_suffix))
        df = run_benchmark(model, pairs, model_name=model_name, output_csv=csv_path)
        df["negator"] = args.negator_suffix
        df.to_csv(csv_path, index=False)
        all_dfs.append(df)

        analysis_pairs = pairs[: min(len(pairs), args.analysis_samples)]
        patch_pairs = analysis_pairs[: min(len(analysis_pairs), args.patching_samples)]

        current_step += 1
        _progress(current_step, total_steps, f"Selecting top inhibition heads for `{model_name}`")
        mean_delta = compute_head_dla_batch(model, analysis_pairs, top_k=args.top_k_heads)
        top_heads = select_top_heads(mean_delta, top_k=args.top_k_heads)

        current_step += 1
        _progress(current_step, total_steps, f"Running amplification sweep for `{model_name}`")
        amp_summary = dataset_amplification_experiment(
            model,
            pairs=analysis_pairs,
            heads=top_heads,
            scales=args.amp_scales,
            verbose=True,
        )

        current_step += 1
        _progress(current_step, total_steps, f"Running activation patching summary for `{model_name}`")
        patch_summary = dataset_activation_patching_experiment(
            model,
            patch_pairs,
            max_samples=None,
            verbose=True,
        )

        head_rows.append(
            {
                "model_name": model_name,
                "analysis_pairs": len(analysis_pairs),
                "top_heads": _top_heads_string(top_heads),
            }
        )
        amp_rows.append(
            {
                "model_name": model_name,
                "analysis_pairs": amp_summary["n_pairs"],
                "baseline_rate": amp_summary["baseline_rate"],
                "best_rate": amp_summary["best_rate"],
                "best_scale": amp_summary["best_scale"],
                "absolute_improvement": amp_summary["absolute_improvement"],
            }
        )
        patch_rows.append(
            {
                "model_name": model_name,
                "patch_samples": patch_summary["n_samples"],
                "best_patch_type": patch_summary["best_overall"]["patch_type"] if patch_summary["best_overall"] else "",
                "best_layer": patch_summary["best_overall"]["layer"] if patch_summary["best_overall"] else np.nan,
                "best_delta": patch_summary["best_overall"]["delta"] if patch_summary["best_overall"] else np.nan,
                "best_resid": f"L{patch_summary['best_layers']['resid']['layer']} ({patch_summary['best_layers']['resid']['delta']:+.3f})",
                "best_mlp": f"L{patch_summary['best_layers']['mlp']['layer']} ({patch_summary['best_layers']['mlp']['delta']:+.3f})",
                "best_attn": f"L{patch_summary['best_layers']['attn']['layer']} ({patch_summary['best_layers']['attn']['delta']:+.3f})",
            }
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    current_step += 1
    _progress(current_step, total_steps, "Combining per-model benchmarks")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(args.results_dir, "all_models_benchmark.csv")
    combined_df.to_csv(combined_csv, index=False)

    current_step += 1
    _progress(current_step, total_steps, "Computing SGR summary tables")
    sgr_summary = analyse_sgr_distribution(combined_df, verbose=False)
    benchmark_df = sgr_summary["benchmark_summary"].sort_values("failure_rate", ascending=False).reset_index(drop=True)
    outcome_df = sgr_summary["outcome_summary"].sort_values(["model_name", "outcome"]).reset_index(drop=True)
    mismatch_df = sgr_summary["edge_cases"].sort_values("model_name").reset_index(drop=True)
    head_df = pd.DataFrame(head_rows).sort_values("model_name").reset_index(drop=True)
    amplification_df = pd.DataFrame(amp_rows).sort_values("model_name").reset_index(drop=True)
    patching_df = pd.DataFrame(patch_rows).sort_values("model_name").reset_index(drop=True)

    current_step += 1
    _progress(current_step, total_steps, "Saving summary CSVs")
    benchmark_df.to_csv(os.path.join(args.results_dir, "benchmark_summary.csv"), index=False)
    outcome_df.to_csv(os.path.join(args.results_dir, "benchmark_outcome_summary.csv"), index=False)
    mismatch_df.to_csv(os.path.join(args.results_dir, "benchmark_edge_cases.csv"), index=False)
    head_df.to_csv(os.path.join(args.results_dir, "head_summary.csv"), index=False)
    amplification_df.to_csv(os.path.join(args.results_dir, "amplification_summary.csv"), index=False)
    patching_df.to_csv(os.path.join(args.results_dir, "patching_summary.csv"), index=False)

    current_step += 1
    _progress(current_step, total_steps, "Extracting supporting findings")
    support_lines = _supportive_findings(benchmark_df, mismatch_df, amplification_df, patching_df)

    current_step += 1
    _progress(current_step, total_steps, "Extracting contradictory findings")
    contradiction_lines = _contradictory_findings(benchmark_df, mismatch_df, amplification_df)

    current_step += 1
    _progress(current_step, total_steps, "Assembling markdown report sections")
    report_lines = [
        "# Cross-Model Experiment Report",
        "",
        "- Models rerun: " + ", ".join(f"`{model_name}`" for model_name in model_names),
        f"- Negator suffix: `{args.negator_suffix}`",
        f"- Benchmark samples per model: `{args.max_samples if args.max_samples >= 0 else 'all available'}`",
        f"- Per-model benchmark CSVs: `{args.results_dir}`",
        f"- Combined benchmark CSV: `{combined_csv}`",
        "",
        "## Benchmark Summary",
        "",
    ]
    report_lines.extend(
        _markdown_table(
            benchmark_df[
                [
                    "model_name",
                    "n_samples",
                    "n_failures",
                    "failure_rate",
                    "median_rank_shift",
                    "median_sgr",
                    "success_median_sgr",
                    "failure_median_sgr",
                ]
            ],
            float_cols={"failure_rate", "median_rank_shift", "median_sgr", "success_median_sgr", "failure_median_sgr"},
        )
    )
    report_lines.extend(
        [
            "Interpretation:",
            "",
            "- `median_rank_shift` is `neg_target_rank - pos_target_rank`; positive values mean negation usually pushes the factual token downward.",
            "- `success_median_sgr` and `failure_median_sgr` show whether the SGR ordering matches the behavioural split.",
            "",
            "## SGR Edge Cases",
            "",
        ]
    )
    report_lines.extend(
        _markdown_table(
            mismatch_df,
            float_cols={"success_mismatch_rate", "failure_mismatch_rate"},
        )
    )
    report_lines.extend(
        [
            "Interpretation:",
            "",
            "- `success_with_sgr_gt1` counts cases where the model suppresses the target even though SGR is above 1.",
            "- `failure_with_sgr_le1` counts cases that go against the simple SGR-threshold story.",
            "",
            "## Head Selection and Interventions",
            "",
        ]
    )
    report_lines.extend(_markdown_table(head_df))
    report_lines.extend(
        _markdown_table(
            amplification_df,
            float_cols={"baseline_rate", "best_rate", "best_scale", "absolute_improvement"},
        )
    )
    report_lines.extend(
        _markdown_table(
            patching_df,
            float_cols={"best_layer", "best_delta"},
        )
    )
    report_lines.extend(
        [
            "## What Helps Our Story",
            "",
        ]
    )
    report_lines.extend([f"- {line}" for line in support_lines] or ["- No strong supporting pattern emerged in this rerun."])
    report_lines.extend(
        [
            "",
            "## What Weakens Our Story",
            "",
        ]
    )
    report_lines.extend([f"- {line}" for line in contradiction_lines] or ["- No obvious contradictory pattern emerged in this rerun."])
    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- No graphs are generated by this report. The evidence is intentionally reduced to tables and concise observations.",
            "- The strongest intervention evidence comes from dataset-level head amplification and activation patching, both run on capped subsets for tractability.",
            "- Pair filtering still depends on single-token targets for each tokenizer, so sample counts differ by model.",
            "",
        ]
    )

    report_path = Path(args.report_md)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    current_step += 1
    _progress(current_step, total_steps, "Writing markdown report to disk")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    current_step += 1
    _progress(current_step, total_steps, "Cross-model experiment run complete")
    print(f"\nSaved cross-model markdown report -> {report_path}")


if __name__ == "__main__":
    main()
