"""
run_cross_model_experiments.py
==============================
Rerun the project experiments across all target models and write a standalone
markdown report with benchmark, amplification, and activation patching
summaries.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.analysis import amplification_sweep, compute_head_dla_batch, dataset_activation_patching_experiment, dataset_amplification_experiment, select_top_heads
from src.analysis.per_head import per_head_dla, plot_head_dla_heatmap
from src.benchmark import analyse_sgr_distribution, run_benchmark
from src.dataset import load_counterfact
from src.metrics import compare_models, negation_failure_rate, sgr_vs_failure_correlation, summary_stats
from src.models import load_model
from src.utils import benchmark_csv_path, dynamic_axis_limits


torch.manual_seed(67)

DEFAULT_MODELS = ["gpt2-small", "pythia-160m"]
CANONICAL_POSITIVE = "The capital of France is"
CANONICAL_NEGATED = "The capital of France is not"
CANONICAL_TARGET = " Paris"


def parse_args():
    p = argparse.ArgumentParser(description="Cross-model rerun with markdown reporting")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to benchmark")
    p.add_argument("--negator_suffix", default=" not", help="Negator suffix used to build prompt pairs")
    p.add_argument("--max_samples", type=int, default=-1, help="CounterFact samples per model (-1 = all)")
    p.add_argument("--analysis_samples", type=int, default=200, help="Samples used for per-head and amplification analysis")
    p.add_argument("--patching_samples", type=int, default=20, help="Samples used for dataset activation patching")
    p.add_argument("--top_k_heads", type=int, default=10, help="Top inhibition heads to amplify")
    p.add_argument("--amp_scales", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 4.0], help="Amplification scales")
    p.add_argument("--results_dir", default="results/cross_model", help="Output directory for CSVs")
    p.add_argument("--fig_dir", default="figures/cross_model", help="Output directory for figures")
    p.add_argument("--report_md", default="reports/cross_model_experiments.md", help="Markdown report path")
    return p.parse_args()

def _plot_cross_model_amplification(amplification_results: dict, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "cross_model_amplification_failure_rate.png")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    all_rates = []

    for model_name, result in amplification_results.items():
        all_rates.extend(result["failure_rates"])
        ax.plot(result["scales"], result["failure_rates"], marker="o", linewidth=2, label=model_name)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, label="baseline")
    ax.set_xlabel("Amplification scale")
    ax.set_ylabel("Negation failure rate")
    ax.set_title("Dataset-level amplification effect across models")
    ax.set_ylim(*dynamic_axis_limits(all_rates, floor=0.0, ceil=1.0))
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_cross_model_patching(patching_results: dict, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "cross_model_activation_patching.png")
    patch_types = ["resid", "mlp", "attn"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    for ax, patch_type in zip(axes, patch_types):
        all_values = []
        for model_name, result in patching_results.items():
            values = result["mean_deltas"][patch_type]
            all_values.extend(values)
            ax.plot(range(len(values)), values, marker="o", linewidth=2, label=model_name)
        ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(f"{patch_type.upper()} patch")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Δ target logit")
        ax.set_ylim(*dynamic_axis_limits(all_values))

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> list[str]:
    if df.empty:
        return ["(no data)", ""]
    float_cols = float_cols or set()
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
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


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    max_samples = None if args.max_samples < 0 else args.max_samples

    all_dfs = []
    per_model = {}

    for model_name in args.models:
        print(f"\n{'#' * 72}")
        print(f"# Running model: {model_name}")
        print(f"{'#' * 72}\n")

        model = load_model(model_name)
        pairs = load_counterfact(max_samples=max_samples, model=model, negator_suffix=args.negator_suffix)
        csv_path = str(benchmark_csv_path(args.results_dir, model_name, args.negator_suffix))
        df = run_benchmark(model, pairs, model_name=model_name, output_csv=csv_path)
        df["negator"] = args.negator_suffix
        df.to_csv(csv_path, index=False)
        all_dfs.append(df)

        analysis_pairs = pairs[:min(len(pairs), args.analysis_samples)]
        patch_pairs = analysis_pairs[:min(len(analysis_pairs), args.patching_samples)]

        mean_delta = compute_head_dla_batch(model, analysis_pairs, top_k=args.top_k_heads)
        top_heads = select_top_heads(mean_delta, top_k=args.top_k_heads)

        head_pos, head_neg = per_head_dla(model, CANONICAL_POSITIVE, CANONICAL_NEGATED, CANONICAL_TARGET)
        headmap_filename = f"{model_name}_head_dla_heatmap.png"
        plot_head_dla_heatmap(
            head_pos,
            head_neg,
            target_token=CANONICAL_TARGET,
            fig_dir=args.fig_dir,
            filename=headmap_filename,
        )

        amp_single = amplification_sweep(
            model,
            positive_prompt=CANONICAL_POSITIVE,
            negated_prompt=CANONICAL_NEGATED,
            target_token=CANONICAL_TARGET,
            heads=top_heads,
            scales=args.amp_scales,
            fig_dir=args.fig_dir,
            filename=f"{model_name}_amplification_sweep.png",
        )
        amp_dataset = dataset_amplification_experiment(
            model,
            pairs=analysis_pairs,
            heads=top_heads,
            scales=args.amp_scales,
            fig_dir=args.fig_dir,
            filename=f"{model_name}_amplification_failure_rate.png",
        )

        patch_summary = dataset_activation_patching_experiment(
            model,
            patch_pairs,
            max_samples=None,
            fig_dir=args.fig_dir,
            filename=f"{model_name}_activation_patching.png",
            title=f"{model_name} activation patching ({len(patch_pairs)} samples)",
        )

        per_model[model_name] = {
            "csv_path": csv_path,
            "n_pairs": len(pairs),
            "analysis_pairs": len(analysis_pairs),
            "patch_pairs": len(patch_pairs),
            "top_heads": top_heads,
            "headmap_path": os.path.join(args.fig_dir, headmap_filename),
            "amp_single": amp_single,
            "amp_dataset": amp_dataset,
            "patching": patch_summary,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(args.results_dir, "all_models_benchmark.csv")
    combined_df.to_csv(combined_csv, index=False)

    analyse_sgr_distribution(combined_df, fig_dir=args.fig_dir)
    summary = summary_stats(combined_df, verbose=False)
    corr_df = sgr_vs_failure_correlation(combined_df)
    fail_df = negation_failure_rate(combined_df)
    comparison = compare_models(combined_df, args.models[0], args.models[1]) if len(args.models) == 2 else None

    amplification_cross_path = _plot_cross_model_amplification(
        {model_name: data["amp_dataset"] for model_name, data in per_model.items()},
        args.fig_dir,
    )
    patching_cross_path = _plot_cross_model_patching(
        {model_name: data["patching"] for model_name, data in per_model.items()},
        args.fig_dir,
    )

    benchmark_rows = []
    for model_name in args.models:
        stats = summary[model_name]
        benchmark_rows.append({
            "model_name": model_name,
            "n_samples": stats["n_samples"],
            "failure_rate": stats["failure_rate"],
            "sgr_mean": stats["sgr_mean"],
            "sgr_median": stats["sgr_median"],
            "sgr_gt1_rate": stats["sgr_gt1_rate"],
            "crossover_present": stats["crossover_present"],
        })
    benchmark_df = pd.DataFrame(benchmark_rows)

    head_rows = []
    amp_rows = []
    patch_rows = []
    for model_name, data in per_model.items():
        head_rows.append({
            "model_name": model_name,
            "top_heads": ", ".join(f"({layer},{head})" for layer, head in data["top_heads"][:5]),
            "head_heatmap": data["headmap_path"],
        })
        amp_dataset = data["amp_dataset"]
        best_idx = int(np.nanargmin(amp_dataset["failure_rates"]))
        amp_rows.append({
            "model_name": model_name,
            "baseline_rate_at_1x": amp_dataset["failure_rates"][amp_dataset["scales"].index(1.0)] if 1.0 in amp_dataset["scales"] else np.nan,
            "best_rate": amp_dataset["failure_rates"][best_idx],
            "best_scale": amp_dataset["scales"][best_idx],
            "failure_curve_figure": amp_dataset["figure_path"],
            "single_prompt_figure": data["amp_single"]["figure_path"],
        })
        patch_rows.append({
            "model_name": model_name,
            "patch_samples": data["patching"]["n_samples"],
            "best_resid": f"L{data['patching']['best_layers']['resid']['layer']} ({data['patching']['best_layers']['resid']['delta']:+.3f})",
            "best_mlp": f"L{data['patching']['best_layers']['mlp']['layer']} ({data['patching']['best_layers']['mlp']['delta']:+.3f})",
            "best_attn": f"L{data['patching']['best_layers']['attn']['layer']} ({data['patching']['best_layers']['attn']['delta']:+.3f})",
            "patch_figure": data["patching"]["figure_path"],
        })

    report_lines = [
        "# Cross-Model Experiment Report",
        "",
        "- Models rerun: " + ", ".join(f"`{m}`" for m in args.models),
        f"- Negator suffix: `{args.negator_suffix}`",
        f"- Benchmark samples per model: `{args.max_samples if args.max_samples >= 0 else 'all available'}`",
        f"- Per-head/amplification sample cap: `{args.analysis_samples}`",
        f"- Activation patching sample cap: `{args.patching_samples}`",
        f"- Combined benchmark CSV: `{combined_csv}`",
        "",
        "## Benchmark Summary",
        "",
    ]
    report_lines.extend(_markdown_table(benchmark_df, float_cols={"failure_rate", "sgr_mean", "sgr_median", "sgr_gt1_rate", "crossover_present"}))
    report_lines.extend([
        "## Correlation Summary",
        "",
    ])
    report_lines.extend(_markdown_table(
        corr_df[["model_name", "n_samples", "spearman_r", "spearman_p", "pointbiserial_r", "pointbiserial_p"]],
        float_cols={"spearman_r", "spearman_p", "pointbiserial_r", "pointbiserial_p"},
    ))
    report_lines.extend([
        "## Failure Rate Confidence Intervals",
        "",
    ])
    report_lines.extend(_markdown_table(fail_df, float_cols={"failure_rate", "ci_lower", "ci_upper"}))
    report_lines.extend([
        "## Per-Head and Amplification Summary",
        "",
    ])
    report_lines.extend(_markdown_table(pd.DataFrame(head_rows)))
    report_lines.extend(_markdown_table(pd.DataFrame(amp_rows), float_cols={"baseline_rate_at_1x", "best_rate", "best_scale"}))
    report_lines.extend([
        f"- Cross-model amplification figure: `{amplification_cross_path}`",
        "",
        "## Activation Patching Summary",
        "",
    ])
    report_lines.extend(_markdown_table(pd.DataFrame(patch_rows)))
    report_lines.extend([
        f"- Cross-model activation patching figure: `{patching_cross_path}`",
        "",
        "## Shared SGR Figures",
        "",
        f"- Histogram: `{os.path.join(args.fig_dir, 'sgr_histogram.png')}`",
        f"- Failure-rate curve: `{os.path.join(args.fig_dir, 'sgr_failure_rate.png')}`",
        f"- Per-layer DLA heatmap: `{os.path.join(args.fig_dir, 'per_layer_dla_mean.png')}`",
        f"- Model comparison: `{os.path.join(args.fig_dir, 'sgr_model_comparison.png')}`" if len(args.models) > 1 else "",
        "",
    ])

    if comparison is not None:
        report_lines.extend([
            "## Two-Model Statistical Comparison",
            "",
            f"- Failure rates: `{args.models[0]}` = {comparison[f'failure_rate_{args.models[0]}']:.4f}, `{args.models[1]}` = {comparison[f'failure_rate_{args.models[1]}']:.4f}",
            f"- Two-proportion z-test: `z = {comparison['two_proportion_z']:.4f}`, `p = {comparison['two_proportion_p']:.4g}`",
            f"- Mann-Whitney on SGR: `U = {comparison['mannwhitney_sgr_u']:.0f}`, `p = {comparison['mannwhitney_sgr_p']:.4g}`",
            "",
        ])

    report_lines.extend([
        "## Notes",
        "",
        f"- SGR plots now use a data-aware range instead of a fixed clip; the current combined run used an automatic cap derived from the plotted data.",
        f"- Amplification and activation-patching figures also use dynamic y-axis ranges based on the actual plotted values.",
        "",
    ])

    report_path = Path(args.report_md)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved cross-model markdown report -> {report_path}")


if __name__ == "__main__":
    main()
