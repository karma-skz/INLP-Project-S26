"""
sgr_lt1_verification.py
========================
SGR < 1 Verification Experiment
--------------------------------
Goal: Confirm that whenever the Signal-to-Gate Ratio (SGR) < 1 (inhibition
outweighs retrieval), the model *consistently* avoids the negation failure,
i.e. it does NOT fall back on the factual token.

This is a *pure analysis* script — it only reads the CSVs produced by the
main pipeline (run_pipeline.py / run_benchmark.py). No model loading needed.

Outputs
-------
  figures/sgr_lt1_verification.png   — dual-panel figure
  figures/sgr_lt1_by_negation.png    — comparison across negation types
  (console)                          — detailed statistics table

Usage
-----
  conda run -n inlp-project python sgr_lt1_verification.py
  python sgr_lt1_verification.py   (if pandas/matplotlib available on PATH)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
FIG_DIR     = "figures"

# Map each CSV to a human-readable negation-type label
DATASETS: list[tuple[str, str]] = [
    ("gpt2-small_benchmark.csv",        'Hard negation ("not")'),
    ("gpt2-small_not_benchmark.csv",    'Soft negation ("is not")'),
    ("gpt2-small_rarely_benchmark.csv", 'Soft negation ("rarely")'),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_df(filename: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, filename)
    df = pd.read_csv(path)

    # Cast booleans
    if df["negation_failure"].dtype == object:
        df["negation_failure"] = df["negation_failure"].map(
            {"True": True, "False": False}
        )

    # Replace inf / str-inf SGR with NaN so we can filter cleanly
    df["sgr"] = pd.to_numeric(df["sgr"], errors="coerce")  # catches "inf" strings
    df["sgr"] = df["sgr"].replace([np.inf, -np.inf], np.nan)

    return df


def sgr_lt1_stats(df: pd.DataFrame, label: str) -> dict:
    """Return a stats dict for the SGR < 1 verification on *df*."""
    valid = df.dropna(subset=["sgr"])
    lt1   = valid[valid["sgr"] < 1]
    ge1   = valid[valid["sgr"] >= 1]

    n_total  = len(valid)
    n_lt1    = len(lt1)
    n_ge1    = len(ge1)
    n_inf    = df["sgr"].isna().sum()   # inf / NaN rows excluded from analysis

    # Among SGR < 1: what fraction are successes (negation_failure == False)?
    success_lt1  = (~lt1["negation_failure"]).sum()
    fail_lt1     = lt1["negation_failure"].sum()
    pct_ok_lt1   = success_lt1 / n_lt1 * 100 if n_lt1 > 0 else float("nan")

    # Among SGR >= 1: what fraction are failures?
    fail_ge1     = ge1["negation_failure"].sum()
    pct_fail_ge1 = fail_ge1 / n_ge1 * 100 if n_ge1 > 0 else float("nan")

    # Overall failure rate
    fail_all     = valid["negation_failure"].sum()
    pct_fail_all = fail_all / n_total * 100 if n_total > 0 else float("nan")

    return {
        "label":        label,
        "n_total":      n_total,
        "n_inf_skipped":n_inf,
        "n_sgr_lt1":    n_lt1,
        "n_sgr_ge1":    n_ge1,
        "success_lt1":  success_lt1,
        "fail_lt1":     fail_lt1,
        "pct_ok_lt1":   pct_ok_lt1,
        "fail_ge1":     fail_ge1,
        "pct_fail_ge1": pct_fail_ge1,
        "fail_all":     fail_all,
        "pct_fail_all": pct_fail_all,
    }


def print_stats_table(stats_list: list[dict]) -> None:
    """Print a neat ASCII table of statistics."""
    header = (
        f"{'Dataset':<38} {'N':>6} {'SGR<1':>7} {'✓ in SGR<1':>12} "
        f"{'% Success':>10} {'✗ in SGR≥1':>12} {'% Fail':>8} {'Overall Fail':>13}"
    )
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for s in stats_list:
        print(
            f"{s['label']:<38} {s['n_total']:>6} {s['n_sgr_lt1']:>7} "
            f"{s['success_lt1']:>12} {s['pct_ok_lt1']:>9.1f}% "
            f"{s['fail_ge1']:>12} {s['pct_fail_ge1']:>7.1f}% "
            f"{s['fail_all']:>11}/{s['n_total']}"
        )
    print(sep)
    print()


# ── Figure 1: dual-panel SGR distribution + bar chart (hard negation) ────────

def plot_main_verification(df: pd.DataFrame, stats: dict, fig_dir: str) -> None:
    """
    Two-panel figure for the main (hard-negation) dataset:
      Left  — SGR distribution split by negation failure (log-scale x)
      Right — Bar chart: success/failure rates by SGR region
    """
    valid = df.dropna(subset=["sgr"])
    fail  = valid[valid["negation_failure"]]
    succ  = valid[~valid["negation_failure"]]
    lt1   = valid[valid["sgr"] < 1]
    ge1   = valid[valid["sgr"] >= 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "SGR < 1 Verification — Competitive Gating Hypothesis",
        fontsize=13, fontweight="bold"
    )

    # ── Left: SGR histogram split by outcome ─────────────────────────────────
    ax = axes[0]
    bins = np.logspace(np.log10(max(valid["sgr"].min(), 1e-3)), np.log10(valid["sgr"].max()), 40)
    ax.hist(succ["sgr"].clip(1e-3), bins=bins, color="steelblue", alpha=0.65,
            label="Negation success (no failure)", edgecolor="white", linewidth=0.3)
    ax.hist(fail["sgr"].clip(1e-3), bins=bins, color="salmon", alpha=0.65,
            label="Negation failure (hallucination)", edgecolor="white", linewidth=0.3)
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, label="SGR = 1 (threshold)")
    ax.set_xscale("log")
    ax.set_xlabel("Signal-to-Gate Ratio (SGR) — log scale", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of SGR by Outcome", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")

    # ── Right: stacked bars per SGR region ───────────────────────────────────
    ax2 = axes[1]
    regions   = ["SGR < 1\n(inhibition wins)", "SGR ≥ 1\n(retrieval wins)"]
    n_success = [int((~lt1["negation_failure"]).sum()), int((~ge1["negation_failure"]).sum())]
    n_fail    = [int(lt1["negation_failure"].sum()),    int(ge1["negation_failure"].sum())]
    n_total_r = [len(lt1), len(ge1)]

    x   = np.arange(len(regions))
    w   = 0.45
    bars_s = ax2.bar(x, n_success, w, label="Success", color="steelblue", alpha=0.85)
    bars_f = ax2.bar(x, n_fail,    w, bottom=n_success, label="Failure", color="salmon", alpha=0.85)

    # Annotate percentages
    for i, (ns, nf, nt) in enumerate(zip(n_success, n_fail, n_total_r)):
        pct_s = ns / nt * 100 if nt > 0 else 0
        pct_f = nf / nt * 100 if nt > 0 else 0
        if ns > 0:
            ax2.text(i, ns / 2, f"{pct_s:.1f}%", ha="center", va="center",
                     fontsize=10, fontweight="bold", color="white")
        if nf > 0:
            ax2.text(i, ns + nf / 2, f"{pct_f:.1f}%", ha="center", va="center",
                     fontsize=10, fontweight="bold", color="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, fontsize=11)
    ax2.set_ylabel("Number of samples", fontsize=11)
    ax2.set_title("Outcome by SGR Region (Hard Negation)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#f8f9fa")

    # Key stats text box
    textstr = (
        f"SGR<1 success rate: {stats['pct_ok_lt1']:.1f}%\n"
        f"SGR≥1 failure rate: {stats['pct_fail_ge1']:.1f}%\n"
        f"Inf-SGR samples excluded: {stats['n_inf_skipped']}"
    )
    props = dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right", bbox=props)

    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "sgr_lt1_verification.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: comparison across negation types ────────────────────────────────

def plot_negation_type_comparison(all_stats: list[dict], fig_dir: str) -> None:
    """Bar chart comparing SGR<1 success rate & overall failure rate across negation types."""
    labels        = [s["label"] for s in all_stats]
    pct_ok_lt1    = [s["pct_ok_lt1"]   for s in all_stats]
    pct_fail_all  = [s["pct_fail_all"] for s in all_stats]
    n_sgr_lt1     = [s["n_sgr_lt1"]    for s in all_stats]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "SGR < 1 Verification — Across Negation Types",
        fontsize=12, fontweight="bold"
    )

    bars1 = ax.bar(x - w / 2, pct_ok_lt1,   w, color="steelblue", alpha=0.85,
                   label="% Success when SGR<1 (ours: should be ~100%)")
    bars2 = ax.bar(x + w / 2, pct_fail_all, w, color="salmon",    alpha=0.85,
                   label="Overall negation failure rate")

    # Annotate with n_sgr_lt1
    for bar, n in zip(bars1, n_sgr_lt1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=100, color="steelblue", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, wrap=True)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(
        '"When SGR < 1, inhibition wins" — hypothesis check per negation type',
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")

    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "sgr_lt1_by_negation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  SGR < 1 Verification — Competitive Gating Hypothesis")
    print("=" * 65)

    all_stats: list[dict] = []
    main_df  = None

    for filename, label in DATASETS:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            warnings.warn(f"File not found, skipping: {path}")
            continue

        df    = load_df(filename)
        stats = sgr_lt1_stats(df, label)
        all_stats.append(stats)

        if main_df is None:          # first file = main (hard negation) dataset
            main_df       = df
            main_stats    = stats

    if not all_stats:
        print("No result files found. Run run_pipeline.py first.")
        return

    print_stats_table(all_stats)

    # ── Core result ──
    s = all_stats[0]  # hard negation
    print("─── Core Finding ───────────────────────────────────────────────")
    print(f"  Hard negation dataset: {s['n_total']} valid samples "
          f"({s['n_inf_skipped']} inf-SGR excluded)")
    print(f"  Samples with SGR < 1  :  {s['n_sgr_lt1']}  "
          f"→  {s['pct_ok_lt1']:.1f}% success rate (expected: ~100%)")
    print(f"  Samples with SGR ≥ 1  :  {s['n_sgr_ge1']}  "
          f"→  {s['pct_fail_ge1']:.1f}% failure rate (expected: >>0%)")
    print(
        "\n  Interpretation: " +
        ("✅ STRONG support for Competitive Gating Hypothesis — "
         "SGR<1 is a near-perfect predictor of negation success."
         if s["pct_ok_lt1"] >= 95
         else "⚠️  Moderate support — check for edge cases in SGR<1 failures.")
    )
    print("────────────────────────────────────────────────────────────────\n")

    # ── Figures ──
    if main_df is not None:
        plot_main_verification(main_df, main_stats, FIG_DIR)
    if len(all_stats) > 1:
        plot_negation_type_comparison(all_stats, FIG_DIR)
    else:
        print("Only one dataset found — skipping multi-negation comparison figure.")

    print("\nDone.")


if __name__ == "__main__":
    main()
