"""
soft_negation_experiment.py
============================
Standalone experiment comparing hard vs soft negation.

Tests the hypothesis from Section 8.2 of the mid-submission report:
  "Soft negation (e.g. 'unlikely to be') should produce weaker inhibition
   than hard negation ('not'), especially in smaller models."

For each model x negator combination, the script:
  1. Builds prompt pairs with the specified negator suffix
  2. Runs the full DLA + SGR benchmark
  3. Saves per-model-per-negator CSVs
  4. Generates comparative figures and a markdown report

Usage
-----
  conda activate inlp-project

  # Quick test (200 samples, 1 model)
  python soft_negation_experiment.py

  # Full run
  python soft_negation_experiment.py \
      --models gpt2-small pythia-160m \
      --max_samples -1

  # Custom negators
  python soft_negation_experiment.py \
      --negators " not" " unlikely to be" " rarely" " maybe"
"""

from __future__ import annotations

import argparse
import gc
import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Monkey patch for transformer_lens Pythia/GPTNeoX loading bug
try:
    from transformers import GPTNeoXConfig
    if not hasattr(GPTNeoXConfig, "rotary_pct"):
        GPTNeoXConfig.rotary_pct = 0.25
except ImportError:
    pass

from src.dataset import load_counterfact
from src.models import load_model, MODEL_SHORTNAMES
from src.benchmark import run_benchmark
from src.utils import benchmark_csv_path, load_benchmark_dataframe, safe_suffix

sns.set_theme(style="whitegrid")
torch.manual_seed(67)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_NEGATORS = [" not", " unlikely to be", " rarely"]
DEFAULT_MODELS = ["gpt2-small"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Soft Negation Experiment: compare negator types"
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Models to benchmark (default: gpt2-small)"
    )
    p.add_argument(
        "--negators", nargs="+", default=DEFAULT_NEGATORS,
        help='Negator suffixes to compare (default: " not" " unlikely to be" " rarely")'
    )
    p.add_argument(
        "--max_samples", type=int, default=200,
        help="Max CounterFact samples per run (-1 = all, default: 200)"
    )
    p.add_argument(
        "--results_dir", default="results/soft_negation",
        help="Output directory for CSVs"
    )
    p.add_argument(
        "--fig_dir", default="figures/soft_negation",
        help="Output directory for figures"
    )
    return p.parse_args()


# ── Stage 1: Benchmark all combinations ──────────────────────────────────────

def run_all_benchmarks(args) -> pd.DataFrame:
    """Run the DLA+SGR benchmark for every (model, negator) pair."""
    model_names = [MODEL_SHORTNAMES.get(m, m) for m in args.models]
    max_s = None if args.max_samples < 0 else args.max_samples
    all_dfs = []

    for model_name in model_names:
        model = load_model(model_name)

        for neg in args.negators:
            tag = f"{model_name} | negator='{neg}'"
            print(f"\n{'#' * 70}")
            print(f"# {tag}")
            print(f"{'#' * 70}\n")

            # Check if CSV already exists (skip recomputation)
            csv_path = str(benchmark_csv_path(args.results_dir, model_name, neg))
            if os.path.exists(csv_path):
                print(f"  CSV already exists, loading: {csv_path}")
                df = load_benchmark_dataframe(csv_path)
            else:
                pairs = load_counterfact(
                    max_samples=max_s, model=model, negator_suffix=neg
                )
                df = run_benchmark(
                    model, pairs,
                    model_name=model_name,
                    output_csv=csv_path,
                )
                df["negator"] = neg
                df.to_csv(csv_path, index=False)

            if "negator" not in df.columns:
                df["negator"] = neg
            all_dfs.append(df)

        # Free GPU memory between models
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure clean types
    combined["sgr"] = pd.to_numeric(combined["sgr"], errors="coerce")
    combined["sgr"] = combined["sgr"].replace([np.inf, -np.inf], np.nan)
    if combined["negation_failure"].dtype == object:
        combined["negation_failure"] = combined["negation_failure"].map(
            {"True": True, "False": False}
        )

    combined_path = os.path.join(args.results_dir, "soft_negation_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined CSV -> {combined_path}")
    return combined


# ── Stage 2: Summary statistics ──────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, negator) combination."""
    rows = []
    for (model, neg), g in df.groupby(["model_name", "negator"]):
        valid = g.dropna(subset=["sgr"])
        rows.append({
            "model": model,
            "negator": neg,
            "n_samples": len(g),
            "n_failures": int(g["negation_failure"].sum()),
            "failure_rate": g["negation_failure"].mean(),
            "mean_sgr": valid["sgr"].mean(),
            "median_sgr": valid["sgr"].median(),
            "sgr_gt1_rate": (valid["sgr"] > 1).mean(),
            "sgr_lt1_count": int((valid["sgr"] < 1).sum()),
            "fail_with_sgrgt1": int(
                g[g["negation_failure"]].dropna(subset=["sgr"])
                .pipe(lambda x: x[x["sgr"] > 1]).shape[0]
            ),
        })
    return pd.DataFrame(rows)


# ── Stage 3: Figures ─────────────────────────────────────────────────────────

def _save(fig, fig_dir, name):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _negator_label(neg: str) -> str:
    """Human-readable label for a negator suffix."""
    s = neg.strip()
    if s == "not":
        return 'Hard ("not")'
    return f'Soft ("{s}")'


def plot_failure_rate_comparison(summary: pd.DataFrame, fig_dir: str):
    """Bar chart: failure rate per negator, grouped by model."""
    summary = summary.copy()
    summary["label"] = summary["negator"].apply(_negator_label)

    fig, ax = plt.subplots(figsize=(max(8, 3 * summary["model"].nunique()), 5))
    sns.barplot(data=summary, x="model", y="failure_rate", hue="label",
                palette="viridis", edgecolor=".3", ax=ax)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Negation Failure Rate", fontsize=11)
    ax.set_title("Negation Failure Rate: Hard vs Soft Negation",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Negator Type", fontsize=9)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    _save(fig, fig_dir, "failure_rate_comparison.png")


def plot_sgr_distribution_comparison(df: pd.DataFrame, fig_dir: str):
    """SGR histogram per negator, faceted by model."""
    df = df.dropna(subset=["sgr"]).copy()
    df["label"] = df["negator"].apply(_negator_label)
    models = df["model_name"].unique()

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5),
                             sharey=True, squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = df[df["model_name"] == model]
        lo = max(sub["sgr"].min(), 1e-2)
        hi = sub["sgr"].quantile(0.99)
        if lo >= hi:
            hi = lo * 10
        bins = np.logspace(np.log10(lo), np.log10(hi), 35)

        for label in sub["label"].unique():
            chunk = sub[sub["label"] == label]
            ax.hist(chunk["sgr"].clip(lo), bins=bins, alpha=0.5,
                    edgecolor="white", linewidth=0.3, label=label)

        ax.axvline(1.0, color="black", ls="--", lw=1.3, label="SGR = 1")
        ax.set_xscale("log")
        ax.set_xlabel("SGR (log)")
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_facecolor("#f8f9fa")

    axes[0][0].set_ylabel("Count")
    fig.suptitle("SGR Distribution by Negation Type",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir, "sgr_distribution_comparison.png")


def plot_median_sgr_comparison(summary: pd.DataFrame, fig_dir: str):
    """Grouped bar chart of median SGR per negator type."""
    summary = summary.copy()
    summary["label"] = summary["negator"].apply(_negator_label)

    fig, ax = plt.subplots(figsize=(max(8, 3 * summary["model"].nunique()), 5))
    sns.barplot(data=summary, x="model", y="median_sgr", hue="label",
                palette="magma", edgecolor=".3", ax=ax)

    ax.axhline(1.0, color="indianred", ls="--", lw=1.2, label="SGR = 1")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Median SGR", fontsize=11)
    ax.set_title("Median SGR: Hard vs Soft Negation\n(Lower = stronger inhibition)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Negator Type", fontsize=9)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    _save(fig, fig_dir, "median_sgr_comparison.png")


def plot_inhibition_strength(df: pd.DataFrame, fig_dir: str):
    """Box plot of inhibition strength (negative attention DLA) per negator."""
    df = df.copy()
    df["label"] = df["negator"].apply(_negator_label)

    if "inhibition_strength" not in df.columns:
        print("  Skipping inhibition strength plot (column missing).")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="model_name", y="inhibition_strength",
                hue="label", palette="coolwarm", showfliers=False, ax=ax)
    ax.set_ylabel("Inhibition Strength\n(total negative attention DLA)", fontsize=10)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title("Inhibition Circuit Strength by Negation Type",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Negator", fontsize=9)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    _save(fig, fig_dir, "inhibition_strength_comparison.png")


def plot_failure_by_sgr_region(df: pd.DataFrame, summary: pd.DataFrame, fig_dir: str):
    """Stacked bar: failure count in SGR<1 vs SGR>=1, per negator."""
    summary = summary.copy()
    summary["label"] = summary["negator"].apply(_negator_label)
    summary["success_sgr_lt1"] = summary["sgr_lt1_count"] - summary.apply(
        lambda r: r["n_failures"] - r["fail_with_sgrgt1"], axis=1
    ).clip(lower=0)

    fig, ax = plt.subplots(figsize=(max(8, 4 * summary["model"].nunique()), 5))
    x = np.arange(len(summary))
    w = 0.6

    ax.bar(x, summary["fail_with_sgrgt1"], w, color="salmon", alpha=0.85,
           label="Failures with SGR ≥ 1")
    ax.bar(x, summary["n_failures"] - summary["fail_with_sgrgt1"], w,
           bottom=summary["fail_with_sgrgt1"], color="lightcoral", alpha=0.5,
           label="Failures with SGR < 1")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['model']}\n{r['label']}" for _, r in summary.iterrows()],
        fontsize=8
    )
    ax.set_ylabel("Number of Failures", fontsize=11)
    ax.set_title("Where Do Failures Occur in SGR Space?",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    _save(fig, fig_dir, "failure_by_sgr_region.png")


# ── Stage 4: Markdown report ─────────────────────────────────────────────────

def write_report(summary: pd.DataFrame, args, fig_dir: str):
    """Generate a self-contained markdown report."""
    report_path = os.path.join(args.results_dir, "soft_negation_report.md")

    lines = [
        "# Soft Negation Experiment Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\n- Models: {', '.join(args.models)}",
        f"- Negators: {args.negators}",
        f"- Max samples per run: {args.max_samples}",
        f"- Results directory: `{args.results_dir}`",
        f"- Figures directory: `{fig_dir}`",
        "",
        "## Summary Table",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Key Findings",
        "",
    ]

    # Auto-generate findings
    for model in summary["model"].unique():
        msub = summary[summary["model"] == model]
        hard = msub[msub["negator"] == " not"]
        softs = msub[msub["negator"] != " not"]

        if hard.empty or softs.empty:
            continue

        hard_fail = hard.iloc[0]["failure_rate"]
        hard_sgr = hard.iloc[0]["median_sgr"]

        lines.append(f"### {model}")
        lines.append("")

        for _, row in softs.iterrows():
            neg_label = _negator_label(row["negator"])
            delta_fail = row["failure_rate"] - hard_fail
            delta_sgr = row["median_sgr"] - hard_sgr
            direction = "higher" if delta_fail > 0 else "lower"
            sgr_dir = "higher" if delta_sgr > 0 else "lower"

            lines.append(
                f"- **{neg_label}**: failure rate={row['failure_rate']:.1%} "
                f"({direction} than hard negation by {abs(delta_fail):.1%}), "
                f"median SGR={row['median_sgr']:.1f} "
                f"({sgr_dir} than hard negation's {hard_sgr:.1f})"
            )

        lines.append("")

    # Hypothesis check
    lines.extend([
        "## Hypothesis Check",
        "",
        "> *Mid-submission prediction*: Soft negation should produce **weaker** "
        "inhibition than hard negation, yielding higher SGR and higher failure rates, "
        "especially in smaller models.",
        "",
    ])

    has_weaker = False
    for model in summary["model"].unique():
        msub = summary[summary["model"] == model]
        hard = msub[msub["negator"] == " not"]
        softs = msub[msub["negator"] != " not"]
        if hard.empty or softs.empty:
            continue
        for _, row in softs.iterrows():
            if row["median_sgr"] > hard.iloc[0]["median_sgr"]:
                has_weaker = True

    if has_weaker:
        lines.append("**Result**: ✅ Confirmed — at least one soft negator produces "
                      "higher median SGR (weaker inhibition) than hard negation.")
    else:
        lines.append("**Result**: ❌ Not confirmed in this run — soft negators did not "
                      "produce consistently weaker inhibition.")

    lines.extend([
        "",
        "## Figures",
        "",
        f"![Failure Rate Comparison]({fig_dir}/failure_rate_comparison.png)",
        f"![SGR Distribution Comparison]({fig_dir}/sgr_distribution_comparison.png)",
        f"![Median SGR Comparison]({fig_dir}/median_sgr_comparison.png)",
        f"![Inhibition Strength]({fig_dir}/inhibition_strength_comparison.png)",
        f"![Failure by SGR Region]({fig_dir}/failure_by_sgr_region.png)",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport -> {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    print("=" * 70)
    print("  SOFT NEGATION EXPERIMENT")
    print(f"  Models   : {args.models}")
    print(f"  Negators : {args.negators}")
    print(f"  Samples  : {args.max_samples}")
    print("=" * 70)

    # Stage 1: Benchmark
    df = run_all_benchmarks(args)

    # Stage 2: Summary stats
    summary = compute_summary(df)
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    # Stage 3: Figures
    print("\n--- Generating figures ---")
    plot_failure_rate_comparison(summary, args.fig_dir)
    plot_sgr_distribution_comparison(df, args.fig_dir)
    plot_median_sgr_comparison(summary, args.fig_dir)
    plot_inhibition_strength(df, args.fig_dir)
    plot_failure_by_sgr_region(df, summary, args.fig_dir)

    # Stage 4: Report
    write_report(summary, args, args.fig_dir)

    print("\n" + "=" * 70)
    print("  SOFT NEGATION EXPERIMENT COMPLETE")
    print(f"  CSVs    -> {args.results_dir}/")
    print(f"  Figures -> {args.fig_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
