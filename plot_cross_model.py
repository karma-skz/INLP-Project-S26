"""
plot_cross_model.py
====================
Offline graph generator for cross-model experiment results.
Reads only the CSVs in results/cross_model/ — no model loading required.

Usage
-----
    python plot_cross_model.py
    python plot_cross_model.py --results_dir results/cross_model --fig_dir figures/cross_model
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate cross-model figures from CSVs.")
    p.add_argument("--results_dir", default="results/cross_model",
                   help="Directory containing the benchmark CSVs.")
    p.add_argument("--fig_dir", default="figures/cross_model",
                   help="Directory to save generated figures.")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_combined(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "all_models_benchmark.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Combined CSV not found: {path}")
    df = pd.read_csv(path)
    if df["negation_failure"].dtype == object:
        df["negation_failure"] = df["negation_failure"].map({"True": True, "False": False})
    df["sgr"] = pd.to_numeric(df["sgr"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def _save(fig, fig_dir: str, name: str):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1: SGR Histogram per model ────────────────────────────────────────

def plot_sgr_histogram(df: pd.DataFrame, fig_dir: str):
    models = df["model_name"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model_name"] == model].dropna(subset=["sgr"])
        succ = sub[~sub["negation_failure"]]
        fail = sub[sub["negation_failure"]]

        lo = max(sub["sgr"].min(), 1e-2)
        hi = sub["sgr"].quantile(0.99)
        bins = np.logspace(np.log10(lo), np.log10(hi), 35)

        ax.hist(succ["sgr"].clip(lo), bins=bins, color="steelblue", alpha=0.65,
                edgecolor="white", linewidth=0.3, label="Success")
        ax.hist(fail["sgr"].clip(lo), bins=bins, color="salmon", alpha=0.65,
                edgecolor="white", linewidth=0.3, label="Failure")
        ax.axvline(1.0, color="black", ls="--", lw=1.3, label="SGR = 1")
        ax.set_xscale("log")
        ax.set_xlabel("SGR (log)")
        ax.set_title(f"{model}\n(n={len(sub)}, fail={len(fail)})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_facecolor("#f8f9fa")

    axes[0].set_ylabel("Count")
    fig.suptitle("SGR Distribution by Model", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir, "sgr_histogram.png")


# ── Figure 2: SGR vs Failure Rate (threshold sweep) ──────────────────────────

def plot_sgr_failure_rate(df: pd.DataFrame, fig_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    models = df["model_name"].unique()
    colors = sns.color_palette("husl", len(models))

    for model, color in zip(models, colors):
        sub = df[df["model_name"] == model].dropna(subset=["sgr"])
        thresholds = np.linspace(0, sub["sgr"].quantile(0.95), 30)
        rates = []
        for t in thresholds:
            above = sub[sub["sgr"] >= t]
            rates.append(above["negation_failure"].mean() * 100 if len(above) > 0 else 0)
        ax.plot(thresholds, rates, marker="o", markersize=3, linewidth=2,
                color=color, label=model)

    ax.set_xlabel("SGR Threshold", fontsize=11)
    ax.set_ylabel("Negation Failure Rate (%)", fontsize=11)
    ax.set_title("Failure Rate vs SGR Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    _save(fig, fig_dir, "sgr_failure_rate.png")


# ── Figure 3: Per-layer DLA heatmap (success vs failure) ─────────────────────

def _parse_dla_str(series: pd.Series) -> np.ndarray:
    """Parse pipe-separated DLA strings into a 2-D array."""
    rows = []
    for s in series:
        if pd.isna(s):
            continue
        vals = [float(x) for x in str(s).split("|")]
        rows.append(vals)
    if not rows:
        return np.array([])
    # Pad to max length
    max_len = max(len(r) for r in rows)
    padded = [r + [0.0] * (max_len - len(r)) for r in rows]
    return np.array(padded)


def plot_per_layer_dla(df: pd.DataFrame, fig_dir: str):
    models = df["model_name"].unique()
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)
    fig.suptitle("Mean Per-Layer DLA on Negated Prompts\n(Left: FFN retrieval, Right: Attn inhibition)",
                 fontsize=12, fontweight="bold", y=1.02)

    for i, model in enumerate(models):
        sub = df[df["model_name"] == model]
        for j, (col, title, cmap) in enumerate([
            ("ffn_dla_neg_str", "FFN (retrieval)", "Reds"),
            ("attn_dla_neg_str", "Attn (inhibition)", "Blues_r"),
        ]):
            ax = axes[i, j]
            if col not in sub.columns or sub[col].dropna().empty:
                ax.set_visible(False)
                continue

            succ_arr = _parse_dla_str(sub[~sub["negation_failure"]][col])
            fail_arr = _parse_dla_str(sub[sub["negation_failure"]][col])

            if succ_arr.size == 0 or fail_arr.size == 0:
                ax.set_visible(False)
                continue

            # Stack: row 0 = success mean, row 1 = failure mean
            heat = np.vstack([succ_arr.mean(axis=0), fail_arr.mean(axis=0)])
            sns.heatmap(heat, ax=ax, cmap=cmap, center=0,
                        yticklabels=["Success", "Failure"],
                        xticklabels=[f"L{l}" for l in range(heat.shape[1])],
                        annot=False, cbar_kws={"shrink": 0.8})
            ax.set_title(f"{model} — {title}", fontsize=9, fontweight="bold")
            ax.tick_params(axis="x", labelsize=7, rotation=0)

    fig.tight_layout()
    _save(fig, fig_dir, "per_layer_dla_mean.png")


# ── Figure 4: Model comparison bar chart ─────────────────────────────────────

def plot_model_comparison(df: pd.DataFrame, fig_dir: str):
    summary = (
        df.groupby("model_name")
        .agg(
            n=("negation_failure", "size"),
            failures=("negation_failure", "sum"),
            failure_rate=("negation_failure", "mean"),
            median_sgr=("sgr", "median"),
        )
        .reset_index()
        .sort_values("failure_rate")
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: failure rate
    ax = axes[0]
    colors = ["steelblue" if m.startswith("gpt") else "mediumorchid"
              for m in summary["model_name"]]
    bars = ax.barh(summary["model_name"], summary["failure_rate"] * 100,
                   color=colors, edgecolor="white", linewidth=0.5)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{row['failure_rate']*100:.1f}%  (n={int(row['n'])})",
                va="center", fontsize=9)
    ax.set_xlabel("Negation Failure Rate (%)", fontsize=11)
    ax.set_title("Failure Rate by Model", fontsize=11, fontweight="bold")
    ax.set_facecolor("#f8f9fa")

    # Right: median SGR
    ax2 = axes[1]
    bars2 = ax2.barh(summary["model_name"], summary["median_sgr"],
                     color=colors, edgecolor="white", linewidth=0.5)
    for bar, (_, row) in zip(bars2, summary.iterrows()):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{row['median_sgr']:.1f}", va="center", fontsize=9)
    ax2.set_xlabel("Median SGR", fontsize=11)
    ax2.set_title("Median SGR by Model", fontsize=11, fontweight="bold")
    ax2.set_facecolor("#f8f9fa")

    fig.suptitle("Cross-Model Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir, "sgr_model_comparison.png")


# ── Figure 5: Crossover layer distribution ───────────────────────────────────

def plot_crossover(df: pd.DataFrame, fig_dir: str):
    valid = df.dropna(subset=["crossover_layer"]).copy()
    if valid.empty:
        print("  Skipping crossover plots (no crossover data).")
        return

    models = valid["model_name"].unique()

    # 5a: histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=valid, x="crossover_layer", hue="model_name",
                 multiple="dodge", palette="viridis",
                 bins=max(1, int(valid["crossover_layer"].max())),
                 edgecolor=".3", linewidth=0.5, ax=ax)
    ax.set_xlabel("Crossover Layer", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of the Logical Crossover Layer", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, fig_dir, "crossover_layer_dist.png")

    # 5b: crossover vs failure
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=valid, x="negation_failure", y="crossover_layer",
                   hue="negation_failure",
                   palette=["mediumseagreen", "indianred"], inner="quartile",
                   legend=False, ax=ax)
    ax.set_xlabel("Negation Failure (Hallucination)", fontsize=11)
    ax.set_ylabel("Crossover Layer", fontsize=11)
    ax.set_title("Does Late Crossover Predict Failure?", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, fig_dir, "crossover_vs_failure.png")

    # 5c: crossover vs SGR
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_valid = valid.dropna(subset=["sgr"])
    sns.scatterplot(data=plot_valid, x="crossover_layer", y="sgr", hue="model_name",
                    palette="viridis", alpha=0.4, s=10, ax=ax)
    ax.set_yscale("log")
    ax.axhline(1.0, color="indianred", ls="--", lw=1.2, label="SGR = 1")
    ax.set_xlabel("Crossover Layer", fontsize=11)
    ax.set_ylabel("SGR (log scale)", fontsize=11)
    ax.set_title("Crossover Layer vs SGR", fontsize=12, fontweight="bold")
    ax.legend(title="Model", fontsize=8)
    fig.tight_layout()
    _save(fig, fig_dir, "crossover_vs_sgr.png")


# ── Figure 6: SGR < 1 verification ───────────────────────────────────────────

def plot_sgr_lt1_verification(df: pd.DataFrame, fig_dir: str):
    models = df["model_name"].unique()
    rows = []
    for model in models:
        sub = df[df["model_name"] == model].dropna(subset=["sgr"])
        lt1 = sub[sub["sgr"] < 1]
        ge1 = sub[sub["sgr"] >= 1]
        rows.append({
            "model": model,
            "n_lt1": len(lt1),
            "success_lt1": int((~lt1["negation_failure"]).sum()),
            "fail_lt1": int(lt1["negation_failure"].sum()),
            "n_ge1": len(ge1),
            "fail_ge1": int(ge1["negation_failure"].sum()),
        })

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rows))
    w = 0.35

    success_lt1 = [r["success_lt1"] for r in rows]
    fail_lt1 = [r["fail_lt1"] for r in rows]
    fail_ge1_pct = [r["fail_ge1"] / r["n_ge1"] * 100 if r["n_ge1"] > 0 else 0 for r in rows]
    labels = [r["model"] for r in rows]

    ax.bar(x - w/2, success_lt1, w, color="steelblue", alpha=0.85, label="SGR<1 Success")
    ax.bar(x - w/2, fail_lt1, w, bottom=success_lt1, color="salmon", alpha=0.85, label="SGR<1 Failure")

    ax2 = ax.twinx()
    ax2.bar(x + w/2, fail_ge1_pct, w, color="lightcoral", alpha=0.5, label="SGR≥1 Fail %")
    ax2.set_ylabel("SGR ≥ 1 Failure Rate (%)", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count (SGR < 1 samples)", fontsize=10)
    ax.set_title("SGR < 1 Verification\nWhen inhibition wins, does the model always succeed?",
                 fontsize=11, fontweight="bold")

    # Annotate
    for i, r in enumerate(rows):
        n = r["n_lt1"]
        pct = r["success_lt1"] / n * 100 if n > 0 else 0
        ax.text(i - w/2, r["success_lt1"] + r["fail_lt1"] + 0.5,
                f"n={n}\n{pct:.0f}% ok", ha="center", fontsize=8)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    _save(fig, fig_dir, "sgr_lt1_verification.png")


# ── Figure 7: Amplification summary ──────────────────────────────────────────

def plot_amplification(results_dir: str, fig_dir: str):
    path = os.path.join(results_dir, "amplification_summary.csv")
    if not os.path.exists(path):
        print("  Skipping amplification plot (no amplification_summary.csv).")
        return

    amp = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 5))

    models = amp["model_name"].unique()
    colors = {"gpt2-small": "steelblue", "pythia-160m": "mediumorchid"}
    markers = {"gpt2-small": "o", "pythia-160m": "s"}

    for model in models:
        row = amp[amp["model_name"] == model].iloc[0]
        baseline = row["baseline_rate"] * 100
        best = row["best_rate"] * 100
        best_scale = row["best_scale"]

        scales = [1.0, best_scale]
        rates = [baseline, best]
        ax.plot(scales, rates, marker=markers.get(model, "^"),
                color=colors.get(model, "gray"), linewidth=2.5, markersize=8,
                label=f"{model} ({baseline:.1f}% → {best:.1f}%)")

    ax.axvline(1.0, color="gray", ls="--", lw=1, label="Baseline")
    ax.set_xlabel(r"Inhibition Head Amplification Scale ($\alpha$)", fontsize=11)
    ax.set_ylabel("Negation Failure Rate (%)", fontsize=11)
    ax.set_title("Amplification Effectiveness\nScaling top-10 inhibition heads",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    _save(fig, fig_dir, "amplification_summary.png")


# ── Figure 8: Patching summary ───────────────────────────────────────────────

def plot_patching(results_dir: str, fig_dir: str):
    path = os.path.join(results_dir, "patching_summary.csv")
    if not os.path.exists(path):
        print("  Skipping patching plot (no patching_summary.csv).")
        return

    patch = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 5))

    models = patch["model_name"].values
    best_deltas = patch["best_delta"].values
    best_types = patch["best_patch_type"].values
    best_layers = patch["best_layer"].astype(int).values

    colors = ["steelblue" if m.startswith("gpt") else "mediumorchid" for m in models]
    bars = ax.bar(models, best_deltas, color=colors, edgecolor="white", linewidth=0.5)

    for bar, btype, blayer, delta in zip(bars, best_types, best_layers, best_deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{btype} L{blayer}\nΔ={delta:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Best Mean Logit Delta (Δ)", fontsize=11)
    ax.set_title("Activation Patching: Best Rescue Effect per Model",
                 fontsize=12, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    fig.tight_layout()
    _save(fig, fig_dir, "patching_summary.png")


# ── Figure 9: Success vs Failure outcome comparison ──────────────────────────

def plot_outcome_comparison(results_dir: str, fig_dir: str):
    path = os.path.join(results_dir, "benchmark_outcome_summary.csv")
    if not os.path.exists(path):
        print("  Skipping outcome comparison (no benchmark_outcome_summary.csv).")
        return

    oc = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: median rank shift by outcome
    ax = axes[0]
    for i, outcome in enumerate(["success", "failure"]):
        sub = oc[oc["outcome"] == outcome]
        x = np.arange(len(sub))
        color = "steelblue" if outcome == "success" else "salmon"
        offset = -0.2 if outcome == "success" else 0.2
        ax.bar(x + offset, sub["median_rank_shift"], 0.35, color=color,
               alpha=0.85, label=outcome.title())
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_name"], fontsize=9)
    ax.set_ylabel("Median Rank Shift", fontsize=10)
    ax.set_title("Median Rank Shift\n(neg_rank − pos_rank)", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")

    # Right: median SGR by outcome
    ax2 = axes[1]
    for i, outcome in enumerate(["success", "failure"]):
        sub = oc[oc["outcome"] == outcome]
        x = np.arange(len(sub))
        color = "steelblue" if outcome == "success" else "salmon"
        offset = -0.2 if outcome == "success" else 0.2
        ax2.bar(x + offset, sub["median_sgr"], 0.35, color=color,
                alpha=0.85, label=outcome.title())
        ax2.set_xticks(x)
        ax2.set_xticklabels(sub["model_name"], fontsize=9)
    ax2.set_ylabel("Median SGR", fontsize=10)
    ax2.set_title("Median SGR by Outcome", fontsize=11, fontweight="bold")
    ax2.axhline(1.0, color="indianred", ls="--", lw=1, label="SGR = 1")
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#f8f9fa")

    fig.suptitle("Success vs Failure: Key Metrics", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir, "outcome_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    print("=" * 60)
    print("  Cross-Model Figure Generator")
    print(f"  Reading : {args.results_dir}")
    print(f"  Writing : {args.fig_dir}")
    print("=" * 60)

    # Load full benchmark
    df = load_combined(args.results_dir)

    print("\n--- Generating figures ---")
    plot_sgr_histogram(df, args.fig_dir)
    plot_sgr_failure_rate(df, args.fig_dir)
    plot_per_layer_dla(df, args.fig_dir)
    plot_model_comparison(df, args.fig_dir)
    plot_crossover(df, args.fig_dir)
    plot_sgr_lt1_verification(df, args.fig_dir)
    plot_amplification(args.results_dir, args.fig_dir)
    plot_patching(args.results_dir, args.fig_dir)
    plot_outcome_comparison(args.results_dir, args.fig_dir)

    print(f"\n{'=' * 60}")
    print(f"  Done! All figures saved to {args.fig_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
