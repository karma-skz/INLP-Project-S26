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
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Regenerate markdown report from existing CSVs in --results_dir without rerunning experiments",
    )
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


def _read_optional_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _build_report_lines(
    *,
    model_names: list[str],
    args: argparse.Namespace,
    combined_csv: str,
    benchmark_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    head_df: pd.DataFrame,
    amplification_df: pd.DataFrame,
    patching_df: pd.DataFrame,
) -> list[str]:
    report_lines = [
        "# Cross-Model Experiment Report",
        "",
        "- Models rerun: " + ", ".join(f"`{model_name}`" for model_name in model_names),
        f"- Negator suffix: `{args.negator_suffix}`",
        f"- Benchmark samples per model: `{args.max_samples if args.max_samples >= 0 else 'all available'}`",
        f"- Per-model benchmark CSVs: `{args.results_dir}`",
        f"- Combined benchmark CSV: `{combined_csv}`",
        "",
        "## Benchmark Summary Metrics",
        "",
    ]
    if not benchmark_df.empty:
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
    else:
        report_lines.extend(["(no data)", ""])

    report_lines.extend(["## Outcome Metrics", ""])
    report_lines.extend(_markdown_table(outcome_df, float_cols={"median_rank_shift", "median_sgr"}))

    report_lines.extend(["## SGR Edge Case Metrics", ""])
    report_lines.extend(_markdown_table(mismatch_df, float_cols={"success_mismatch_rate", "failure_mismatch_rate"}))

    report_lines.extend(["## Head Selection Metrics", ""])
    report_lines.extend(_markdown_table(head_df))

    report_lines.extend(["## Amplification Metrics", ""])
    report_lines.extend(
        _markdown_table(
            amplification_df,
            float_cols={"baseline_rate", "best_rate", "best_scale", "absolute_improvement"},
        )
    )

    report_lines.extend(["## Activation Patching Metrics", ""])
    report_lines.extend(_markdown_table(patching_df, float_cols={"best_layer", "best_delta"}))

    return report_lines


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    model_names = [MODEL_SHORTNAMES.get(model_name, model_name) for model_name in args.models]
    max_samples = None if args.max_samples < 0 else args.max_samples

    if args.report_only:
        combined_csv = os.path.join(args.results_dir, "all_models_benchmark.csv")

        benchmark_df = _read_optional_csv(os.path.join(args.results_dir, "benchmark_summary.csv"))
        outcome_df = _read_optional_csv(os.path.join(args.results_dir, "benchmark_outcome_summary.csv"))
        mismatch_df = _read_optional_csv(os.path.join(args.results_dir, "benchmark_edge_cases.csv"))
        head_df = _read_optional_csv(os.path.join(args.results_dir, "head_summary.csv"))
        amplification_df = _read_optional_csv(os.path.join(args.results_dir, "amplification_summary.csv"))
        patching_df = _read_optional_csv(os.path.join(args.results_dir, "patching_summary.csv"))

        if (benchmark_df.empty or outcome_df.empty or mismatch_df.empty) and os.path.exists(combined_csv):
            combined_df = pd.read_csv(combined_csv)
            sgr_summary = analyse_sgr_distribution(combined_df, verbose=False)
            benchmark_df = sgr_summary["benchmark_summary"].sort_values("failure_rate", ascending=False).reset_index(drop=True)
            outcome_df = sgr_summary["outcome_summary"].sort_values(["model_name", "outcome"]).reset_index(drop=True)
            mismatch_df = sgr_summary["edge_cases"].sort_values("model_name").reset_index(drop=True)

        report_lines = _build_report_lines(
            model_names=model_names,
            args=args,
            combined_csv=combined_csv,
            benchmark_df=benchmark_df,
            outcome_df=outcome_df,
            mismatch_df=mismatch_df,
            head_df=head_df,
            amplification_df=amplification_df,
            patching_df=patching_df,
        )

        report_path = Path(args.report_md)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Saved cross-model markdown report -> {report_path}")
        return

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
    _progress(current_step, total_steps, "Assembling markdown report sections")
    report_lines = _build_report_lines(
        model_names=model_names,
        args=args,
        combined_csv=combined_csv,
        benchmark_df=benchmark_df,
        outcome_df=outcome_df,
        mismatch_df=mismatch_df,
        head_df=head_df,
        amplification_df=amplification_df,
        patching_df=patching_df,
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
