from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import dynamic_axis_limits, load_benchmark_dataframe


sns.set_theme(style="whitegrid")
matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse crossover-layer behaviour from benchmark CSVs")
    parser.add_argument("--results_csv", default="results/all_models_benchmark.csv", help="Benchmark CSV to analyse")
    parser.add_argument("--fig_dir", default="figures", help="Directory to save figures into")
    return parser.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Please run run_pipeline.py first.")

    df = load_benchmark_dataframe(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    df_valid = df.dropna(subset=["crossover_layer"]).copy()
    print(f"Retained {len(df_valid)} samples with a valid crossover layer.")
    return df_valid


def plot_crossover_histogram(df: pd.DataFrame, fig_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="crossover_layer",
        hue="model_name",
        multiple="dodge",
        bins=max(1, int(df["crossover_layer"].max())),
        palette="viridis",
        edgecolor=".3",
        linewidth=.5,
        ax=ax,
    )
    ax.set_xlabel("Crossover Layer", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of the Logical Crossover Layer", fontsize=12, fontweight="bold")

    try:
        ax.set_xticks(range(int(df["crossover_layer"].min()), int(df["crossover_layer"].max()) + 1))
    except ValueError:
        pass

    out_path = os.path.join(fig_dir, "crossover_layer_dist.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_crossover_vs_failure(df: pd.DataFrame, fig_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.violinplot(
        data=df,
        x="negation_failure",
        y="crossover_layer",
        hue="negation_failure",
        palette=["mediumseagreen", "indianred"],
        inner="quartile",
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("Negation Failure (Hallucination)", fontsize=11)
    ax.set_ylabel("Crossover Layer", fontsize=11)
    ax.set_title("Does Late Crossover Predict Failure?", fontsize=12, fontweight="bold")
    ax.set_ylim(*dynamic_axis_limits(df["crossover_layer"], floor=0.0))

    out_path = os.path.join(fig_dir, "crossover_vs_failure.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_crossover_vs_sgr(df: pd.DataFrame, fig_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="crossover_layer",
        y="sgr",
        hue="model_name",
        palette="viridis",
        alpha=0.7,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.axhline(1.0, color="indianred", linestyle="--", label="SGR = 1")

    ax.set_xlabel("Crossover Layer", fontsize=11)
    ax.set_ylabel("SGR (log scale)", fontsize=11)
    ax.set_title("Crossover Layer vs SGR", fontsize=12, fontweight="bold")
    ax.set_xlim(*dynamic_axis_limits(df["crossover_layer"], floor=0.0))
    ax.legend(title="Model")

    out_path = os.path.join(fig_dir, "crossover_vs_sgr.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def run_crossover_analysis(results_csv: str = "results/all_models_benchmark.csv", fig_dir: str = "figures") -> list[str]:
    os.makedirs(fig_dir, exist_ok=True)
    df = load_data(results_csv)

    print("\n--- Generating Crossover Analysis Plots ---")
    paths = [
        plot_crossover_histogram(df, fig_dir),
        plot_crossover_vs_failure(df, fig_dir),
        plot_crossover_vs_sgr(df, fig_dir),
    ]
    print("-------------------------------------------\n")
    return paths


def main():
    args = parse_args()
    try:
        run_crossover_analysis(results_csv=args.results_csv, fig_dir=args.fig_dir)
    except FileNotFoundError as exc:
        print(exc)


if __name__ == "__main__":
    main()
