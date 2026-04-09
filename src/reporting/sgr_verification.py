from __future__ import annotations

import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import benchmark_csv_path, load_benchmark_dataframe, resolve_benchmark_csv


matplotlib.use("Agg")


DEFAULT_DATASETS: list[tuple[str, str, str]] = [
    ("gpt2-small", " not", 'Negator suffix "not"'),
    ("gpt2-small", " unlikely to be", 'Negator suffix "unlikely to be"'),
    ("gpt2-small", " rarely", 'Negator suffix "rarely"'),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Verify the SGR<1 hypothesis from benchmark CSVs")
    parser.add_argument("--results_dir", default="results", help="Directory containing benchmark CSVs")
    parser.add_argument("--fig_dir", default="figures", help="Directory to save figures into")
    return parser.parse_args()


def sgr_lt1_stats(df: pd.DataFrame, label: str) -> dict:
    valid = df.dropna(subset=["sgr"])
    lt1 = valid[valid["sgr"] < 1]
    ge1 = valid[valid["sgr"] >= 1]

    n_total = len(valid)
    n_lt1 = len(lt1)
    n_ge1 = len(ge1)
    n_inf = len(df) - len(valid)

    success_lt1 = (~lt1["negation_failure"]).sum()
    fail_lt1 = lt1["negation_failure"].sum()
    pct_ok_lt1 = success_lt1 / n_lt1 * 100 if n_lt1 > 0 else float("nan")

    fail_ge1 = ge1["negation_failure"].sum()
    pct_fail_ge1 = fail_ge1 / n_ge1 * 100 if n_ge1 > 0 else float("nan")

    fail_all = valid["negation_failure"].sum()
    pct_fail_all = fail_all / n_total * 100 if n_total > 0 else float("nan")

    return {
        "label": label,
        "n_total": n_total,
        "n_inf_skipped": n_inf,
        "n_sgr_lt1": n_lt1,
        "n_sgr_ge1": n_ge1,
        "success_lt1": int(success_lt1),
        "fail_lt1": int(fail_lt1),
        "pct_ok_lt1": pct_ok_lt1,
        "fail_ge1": int(fail_ge1),
        "pct_fail_ge1": pct_fail_ge1,
        "fail_all": int(fail_all),
        "pct_fail_all": pct_fail_all,
    }


def print_stats_table(stats_list: list[dict]) -> None:
    header = (
        f"{'Dataset':<34} {'N':>6} {'SGR<1':>7} {'Successes':>10} "
        f"{'% Success':>10} {'Failures@SGR>=1':>16} {'% Fail':>8} {'Overall Fail':>13}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for stats in stats_list:
        print(
            f"{stats['label']:<34} {stats['n_total']:>6} {stats['n_sgr_lt1']:>7} "
            f"{stats['success_lt1']:>10} {stats['pct_ok_lt1']:>9.1f}% "
            f"{stats['fail_ge1']:>16} {stats['pct_fail_ge1']:>7.1f}% "
            f"{stats['fail_all']:>11}/{stats['n_total']}"
        )
    print(separator)
    print()


def plot_main_verification(df: pd.DataFrame, stats: dict, fig_dir: str) -> str:
    valid = df.dropna(subset=["sgr"])
    fail = valid[valid["negation_failure"]]
    succ = valid[~valid["negation_failure"]]
    lt1 = valid[valid["sgr"] < 1]
    ge1 = valid[valid["sgr"] >= 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SGR < 1 Verification — Competitive Gating Hypothesis", fontsize=13, fontweight="bold")

    ax = axes[0]
    bins = np.logspace(np.log10(max(valid["sgr"].min(), 1e-3)), np.log10(valid["sgr"].max()), 40)
    ax.hist(
        succ["sgr"].clip(1e-3),
        bins=bins,
        color="steelblue",
        alpha=0.65,
        label="Negation success",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.hist(
        fail["sgr"].clip(1e-3),
        bins=bins,
        color="salmon",
        alpha=0.65,
        label="Negation failure",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, label="SGR = 1")
    ax.set_xscale("log")
    ax.set_xlabel("Signal-to-Gate Ratio (log scale)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of SGR by Outcome", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")

    ax2 = axes[1]
    regions = ["SGR < 1\n(inhibition wins)", "SGR ≥ 1\n(retrieval wins)"]
    n_success = [int((~lt1["negation_failure"]).sum()), int((~ge1["negation_failure"]).sum())]
    n_fail = [int(lt1["negation_failure"].sum()), int(ge1["negation_failure"].sum())]
    n_total_r = [len(lt1), len(ge1)]

    x = np.arange(len(regions))
    width = 0.45
    ax2.bar(x, n_success, width, label="Success", color="steelblue", alpha=0.85)
    ax2.bar(x, n_fail, width, bottom=n_success, label="Failure", color="salmon", alpha=0.85)

    for idx, (n_s, n_f, n_t) in enumerate(zip(n_success, n_fail, n_total_r)):
        pct_s = n_s / n_t * 100 if n_t > 0 else 0.0
        pct_f = n_f / n_t * 100 if n_t > 0 else 0.0
        if n_s > 0:
            ax2.text(idx, n_s / 2, f"{pct_s:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        if n_f > 0:
            ax2.text(idx, n_s + n_f / 2, f"{pct_f:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, fontsize=11)
    ax2.set_ylabel("Number of samples", fontsize=11)
    ax2.set_title("Outcome by SGR Region", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#f8f9fa")

    text_box = (
        f"SGR<1 success rate: {stats['pct_ok_lt1']:.1f}%\n"
        f"SGR≥1 failure rate: {stats['pct_fail_ge1']:.1f}%\n"
        f"Inf/NaN SGR rows excluded: {stats['n_inf_skipped']}"
    )
    ax2.text(
        0.97,
        0.97,
        text_box,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )

    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "sgr_lt1_verification.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_negation_type_comparison(all_stats: list[dict], fig_dir: str) -> str:
    labels = [stats["label"] for stats in all_stats]
    pct_ok_lt1 = [stats["pct_ok_lt1"] for stats in all_stats]
    pct_fail_all = [stats["pct_fail_all"] for stats in all_stats]
    n_sgr_lt1 = [stats["n_sgr_lt1"] for stats in all_stats]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("SGR < 1 Verification — Across Negation Types", fontsize=12, fontweight="bold")

    bars1 = ax.bar(
        x - width / 2,
        pct_ok_lt1,
        width,
        color="steelblue",
        alpha=0.85,
        label="% Success when SGR<1",
    )
    ax.bar(
        x + width / 2,
        pct_fail_all,
        width,
        color="salmon",
        alpha=0.85,
        label="Overall negation failure rate",
    )

    for bar, count in zip(bars1, n_sgr_lt1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"n={count}", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=100, color="steelblue", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, wrap=True)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title('"When SGR < 1, inhibition wins" — hypothesis check per negator', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")

    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "sgr_lt1_by_negation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def run_sgr_verification(results_dir: str = "results", fig_dir: str = "figures") -> list[str]:
    print("=" * 65)
    print("  SGR < 1 Verification — Competitive Gating Hypothesis")
    print("=" * 65)

    all_stats: list[dict] = []
    main_df = None
    main_stats = None

    for model_name, negator_suffix, label in DEFAULT_DATASETS:
        path = resolve_benchmark_csv(results_dir, model_name, negator_suffix)
        if not path.exists():
            warnings.warn(f"File not found, skipping: {benchmark_csv_path(results_dir, model_name, negator_suffix)}")
            continue

        df = load_benchmark_dataframe(path)
        stats = sgr_lt1_stats(df, label)
        all_stats.append(stats)

        if main_df is None:
            main_df = df
            main_stats = stats

    if not all_stats:
        raise FileNotFoundError("No result files found. Run run_pipeline.py first.")

    print_stats_table(all_stats)

    primary = all_stats[0]
    print("--- Core Finding ------------------------------------------------")
    print(f"  Primary dataset: {primary['n_total']} valid samples ({primary['n_inf_skipped']} inf/NaN excluded)")
    print(f"  Samples with SGR < 1 : {primary['n_sgr_lt1']} -> {primary['pct_ok_lt1']:.1f}% success rate")
    print(f"  Samples with SGR >= 1: {primary['n_sgr_ge1']} -> {primary['pct_fail_ge1']:.1f}% failure rate")
    print("----------------------------------------------------------------\n")

    paths = []
    if main_df is not None and main_stats is not None:
        paths.append(plot_main_verification(main_df, main_stats, fig_dir))
    if len(all_stats) > 1:
        paths.append(plot_negation_type_comparison(all_stats, fig_dir))
    else:
        print("Only one dataset found — skipping multi-negator comparison figure.")

    print("\nDone.")
    return paths


def main():
    args = parse_args()
    try:
        run_sgr_verification(results_dir=args.results_dir, fig_dir=args.fig_dir)
    except FileNotFoundError as exc:
        print(exc)


if __name__ == "__main__":
    main()
