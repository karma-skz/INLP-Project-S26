"""
crossover_analysis.py
=====================
Analyses the "crossover layer" — the specific transformer layer where the
Inhibition (Attn) signal overtakes the Retrieval (FFN) signal — and tests
for correlations with the Signal-to-Gate Ratio (SGR) and logic failure rates.

Usage
-----
    conda run -n inlp-project python crossover_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Basic aesthetic configuration
sns.set_theme(style="whitegrid")
matplotlib.use("Agg")


def load_data(csv_path: str = "results/all_models_benchmark.csv"):
    """Load benchmark data and filter out missing crossover layers."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Please run run_pipeline.py first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # Not all samples crossover (e.g. if FFN >> Attn consistently to the end, or vice versa)
    df_valid = df.dropna(subset=["crossover_layer"]).copy()
    print(f"Retained {len(df_valid)} samples with a valid crossover layer.")
    return df_valid


def plot_crossover_histogram(df: pd.DataFrame, fig_dir: str):
    """Plot the distribution of crossover layers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.histplot(
        data=df, x="crossover_layer", hue="model_name",
        multiple="dodge", bins=max(1, int(df["crossover_layer"].max())),
        palette="viridis", edgecolor=".3", linewidth=.5, ax=ax
    )
    ax.set_xlabel("Crossover Layer", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of the Logical Crossover Layer", fontsize=12, fontweight="bold")
    
    try:
        # Avoid warnings on older seaborn
        ax.set_xticks(range(int(df["crossover_layer"].min()), int(df["crossover_layer"].max()) + 1))
    except ValueError:
        pass
        
    out_path = os.path.join(fig_dir, "crossover_layer_dist.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_crossover_vs_failure(df: pd.DataFrame, fig_dir: str):
    """Correlate the crossover layer with whether the model hallucinated."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Map colours as a list ordered by custom sort
    palette_list = [plt.cm.RdYlGn_r(0), plt.cm.RdYlGn_r(255)] # True=Fail(red), False=Success(green)
    
    sns.violinplot(
        data=df, x="negation_failure", y="crossover_layer", hue="negation_failure",
        palette=["mediumseagreen", "indianred"], inner="quartile",
        legend=False, ax=ax
    )
    
    ax.set_xlabel("Negation Failure (Hallucination)", fontsize=11)
    ax.set_ylabel("Crossover Layer", fontsize=11)
    ax.set_title("Does Late Crossover Predict Failure?", fontsize=12, fontweight="bold")
    
    out_path = os.path.join(fig_dir, "crossover_vs_failure.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_crossover_vs_sgr(df: pd.DataFrame, fig_dir: str):
    """Correlate the crossover layer with the Signal-to-Gate ratio."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.scatterplot(
        data=df, x="crossover_layer", y="sgr", hue="model_name",
        palette="viridis", alpha=0.7, ax=ax
    )
    ax.set_yscale("log")
    ax.axhline(1.0, color="indianred", linestyle="--", label="SGR = 1")
    
    ax.set_xlabel("Crossover Layer", fontsize=11)
    ax.set_ylabel("SGR (log scale)", fontsize=11)
    ax.set_title("Crossover Layer vs SGR", fontsize=12, fontweight="bold")
    ax.legend(title="Model")
    
    out_path = os.path.join(fig_dir, "crossover_vs_sgr.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(e)
        return
        
    print("\n--- Generating Crossover Analysis Plots ---")
    plot_crossover_histogram(df, fig_dir)
    plot_crossover_vs_failure(df, fig_dir)
    plot_crossover_vs_sgr(df, fig_dir)
    print("-------------------------------------------\n")

if __name__ == "__main__":
    main()
