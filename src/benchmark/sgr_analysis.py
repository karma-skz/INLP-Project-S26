"""
src/benchmark/sgr_analysis.py
===============================
Analysis and visualisation of the Signal-to-Gate Ratio (SGR) distribution
over the full CounterFact benchmark.

Produces
--------
  figures/sgr_histogram.png         — SGR distribution, colour-coded by failure
  figures/sgr_failure_rate.png      — failure rate as a function of SGR threshold
  figures/sgr_model_comparison.png  — GPT-2 vs Pythia (when both present)
  figures/per_layer_dla_mean.png    — mean per-layer FFN / Attn DLA heatmap

Usage
-----
    from src.benchmark import analyse_sgr_distribution
    import pandas as pd

    df = pd.read_csv("results/gpt2_benchmark.csv")
    analyse_sgr_distribution(df, fig_dir="figures")

    # Or combine two model CSVs:
    df2 = pd.read_csv("results/pythia_benchmark.csv")
    analyse_sgr_distribution(pd.concat([df, df2]), fig_dir="figures")
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_layer_arr(series: pd.Series) -> np.ndarray:
    """
    Convert a column of pipe-separated strings to a 2-D numpy array.
    Shape: (n_samples, n_layers).
    """
    return np.vstack(
        series.apply(lambda s: np.array([float(x) for x in str(s).split("|")]))
    )


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyse_sgr_distribution(
    df: pd.DataFrame,
    fig_dir: str = "figures",
    sgr_clip: float = 10.0,
    verbose: bool = True,
) -> dict:
    """
    Analyse the SGR distribution and generate publication-quality figures.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`~src.benchmark.run_benchmark.run_benchmark`
        (one row per prompt pair).  May contain multiple models in the
        ``model_name`` column.
    fig_dir : str
        Directory to write figures into.
    sgr_clip : float
        Cap SGR at this value for visualisation (avoids inf dominating axes).
    verbose : bool
        Print summary statistics.

    Returns
    -------
    dict with keys ``failure_rate``, ``mean_sgr``, ``median_sgr``
    """
    os.makedirs(fig_dir, exist_ok=True)

    # ── Clean up ─────────────────────────────────────────────────────────────
    df = df.copy()
    df["sgr_clipped"] = df["sgr"].replace([float("inf"), float("nan")], sgr_clip)
    df["sgr_clipped"] = df["sgr_clipped"].clip(upper=sgr_clip)
    df["negation_failure"] = df["negation_failure"].astype(bool)

    models = df["model_name"].unique().tolist() if "model_name" in df.columns else ["model"]

    if verbose:
        print(f"\n{'='*60}")
        print("SGR Distribution Analysis")
        print(f"{'='*60}")
        for m in models:
            sub = df[df["model_name"] == m] if "model_name" in df.columns else df
            n = len(sub)
            nf = sub["negation_failure"].sum()
            sgr_finite = sub["sgr"][sub["sgr"] != float("inf")]
            print(f"\n  Model : {m}")
            print(f"  Samples          : {n}")
            print(f"  Negation failures: {nf}  ({nf/n:.1%})")
            print(f"  Mean SGR         : {sgr_finite.mean():.3f}")
            print(f"  Median SGR       : {sgr_finite.median():.3f}")
            print(f"  SGR > 1 (failure): {(sgr_finite > 1).mean():.1%}")

    # ── Figure A: SGR histogram ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, colour in zip(models, ["steelblue", "salmon", "seagreen"]):
        sub = df[df["model_name"] == m] if "model_name" in df.columns else df
        success = sub[~sub["negation_failure"]]["sgr_clipped"]
        failure = sub[sub["negation_failure"]]["sgr_clipped"]
        ax.hist(success, bins=40, alpha=0.6, color=colour,
                label=f"{m} — success (SGR≤1)")
        ax.hist(failure, bins=40, alpha=0.6, color="tomato",
                label=f"{m} — failure (SGR>1)", hatch="//")

    ax.axvline(x=1.0, color="black", linewidth=1.5, linestyle="--", label="SGR = 1")
    ax.set_xlabel("Signal-to-Gate Ratio (SGR)")
    ax.set_ylabel("Count")
    ax.set_title("SGR Distribution — Success vs Negation Failure")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "sgr_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    if verbose:
        print(f"\n  Saved {path}")

    # ── Figure B: failure rate vs SGR threshold ───────────────────────────────
    thresholds = np.linspace(0, sgr_clip, 200)
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, colour in zip(models, ["steelblue", "salmon"]):
        sub = df[df["model_name"] == m] if "model_name" in df.columns else df
        rates = [
            sub[sub["sgr_clipped"] <= t]["negation_failure"].mean()
            for t in thresholds
        ]
        ax.plot(thresholds, rates, color=colour, linewidth=2, label=m)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, label="SGR = 1")
    ax.set_xlabel("SGR threshold")
    ax.set_ylabel("Negation failure rate (proportion)")
    ax.set_title("Failure Rate vs SGR Threshold")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(fig_dir, "sgr_failure_rate.png")
    plt.savefig(path, dpi=150)
    plt.close()
    if verbose:
        print(f"  Saved {path}")

    # ── Figure C: model comparison box plot (if >1 model) ─────────────────────
    if len(models) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Box plot of SGR
        data_to_plot = [
            df[df["model_name"] == m]["sgr_clipped"].values for m in models
        ]
        axes[0].boxplot(data_to_plot, labels=models, notch=True, patch_artist=True)
        axes[0].axhline(y=1.0, color="red", linestyle="--")
        axes[0].set_ylabel("SGR (clipped)")
        axes[0].set_title("SGR Distribution by Model")

        # Failure rate bar chart
        failure_rates = [
            df[df["model_name"] == m]["negation_failure"].mean() for m in models
        ]
        axes[1].bar(models, failure_rates, color=["steelblue", "salmon"])
        axes[1].set_ylabel("Negation failure rate")
        axes[1].set_title("Failure Rate by Model")
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        path = os.path.join(fig_dir, "sgr_model_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        if verbose:
            print(f"  Saved {path}")

    # ── Figure D: mean per-layer DLA heatmap ─────────────────────────────────
    # One row per model, columns = layers; two panels (FFN, Attn)
    fig, axes = plt.subplots(len(models), 2,
                              figsize=(14, 3 * len(models) + 1),
                              squeeze=False)
    for row_i, m in enumerate(models):
        sub = df[df["model_name"] == m] if "model_name" in df.columns else df
        if "ffn_dla_neg_str" not in sub.columns:
            continue
        ffn_mat  = _parse_layer_arr(sub["ffn_dla_neg_str"])
        attn_mat = _parse_layer_arr(sub["attn_dla_neg_str"])
        n_layers = ffn_mat.shape[1]

        # Separate failure / success for each layer
        fail_mask    = sub["negation_failure"].values.astype(bool)
        mean_ffn_f   = ffn_mat[fail_mask].mean(axis=0)
        mean_ffn_s   = ffn_mat[~fail_mask].mean(axis=0)
        mean_attn_f  = attn_mat[fail_mask].mean(axis=0)
        mean_attn_s  = attn_mat[~fail_mask].mean(axis=0)

        stacked_ffn  = np.vstack([mean_ffn_s, mean_ffn_f])
        stacked_attn = np.vstack([mean_attn_s, mean_attn_f])

        cmap = "RdBu_r"
        vmax = max(np.abs(stacked_ffn).max(), np.abs(stacked_attn).max(), 0.1)

        sns.heatmap(stacked_ffn, ax=axes[row_i][0], center=0, vmin=-vmax, vmax=vmax,
                    cmap=cmap, xticklabels=range(n_layers),
                    yticklabels=["success", "failure"],
                    cbar_kws={"shrink": 0.6})
        axes[row_i][0].set_title(f"{m} — FFN DLA (negated)")
        axes[row_i][0].set_xlabel("Layer")

        sns.heatmap(stacked_attn, ax=axes[row_i][1], center=0, vmin=-vmax, vmax=vmax,
                    cmap=cmap, xticklabels=range(n_layers),
                    yticklabels=["success", "failure"],
                    cbar_kws={"shrink": 0.6})
        axes[row_i][1].set_title(f"{m} — Attn DLA (negated)")
        axes[row_i][1].set_xlabel("Layer")

    plt.suptitle("Mean Per-Layer DLA on Negated Prompts")
    plt.tight_layout()
    path = os.path.join(fig_dir, "per_layer_dla_mean.png")
    plt.savefig(path, dpi=150)
    plt.close()
    if verbose:
        print(f"  Saved {path}")

    # ── Summary dict ─────────────────────────────────────────────────────────
    sgr_f = df["sgr"][df["sgr"] != float("inf")]
    return {
        "failure_rate": float(df["negation_failure"].mean()),
        "mean_sgr":     float(sgr_f.mean()),
        "median_sgr":   float(sgr_f.median()),
    }
