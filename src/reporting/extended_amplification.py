from __future__ import annotations

import argparse
import gc
import os

import matplotlib
import matplotlib.pyplot as plt
import torch

from src.analysis import compute_head_dla_batch, dataset_amplification_experiment
from src.analysis.per_head import select_top_heads
from src.dataset import load_counterfact
from src.models import load_model
from src.utils import dynamic_axis_limits


matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-model extended amplification sweep")
    parser.add_argument("--models", nargs="+", default=["gpt2-small", "pythia-160m"], help="Models to evaluate")
    parser.add_argument("--max_samples", type=int, default=200, help="Number of dataset samples per model")
    parser.add_argument("--top_k", type=int, default=10, help="Number of inhibition heads to amplify")
    parser.add_argument("--fig_dir", default="figures", help="Directory to save figures into")
    return parser.parse_args()


def plot_failure_curves(model_failure_rates: dict[str, list[float]], scales: list[float], top_k: int, fig_dir: str) -> str | None:
    if not model_failure_rates:
        print("No valid results computed.")
        return None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"gpt2-small": "steelblue", "pythia-160m": "mediumorchid"}
    markers = {"gpt2-small": "o", "pythia-160m": "s"}

    percent_values = []
    for model_name, failure_rates in model_failure_rates.items():
        percentages = [rate * 100 for rate in failure_rates]
        percent_values.extend(percentages)
        ax.plot(
            scales,
            percentages,
            marker=markers.get(model_name, "o"),
            color=colors.get(model_name, "slategray"),
            linewidth=2.5,
            markersize=7,
            label=f"{model_name} (Top {top_k} heads)",
        )

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.2, label="Baseline (scale=1.0)")
    ax.set_xlabel(r"Inhibition Head Amplification Scale ($\alpha$)", fontsize=11)
    ax.set_ylabel("Negation Failure Rate (%)", fontsize=11)
    ax.set_title("Cross-Model Amplification: Pythia vs GPT-2", fontsize=12, fontweight="bold")
    ax.set_ylim(*dynamic_axis_limits(percent_values, floor=0.0, ceil=100.0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    ax.set_facecolor("#f8f9fa")

    plt.tight_layout()
    output_path = os.path.join(fig_dir, "extended_amplification.png")
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Success! Final comparative graph saved to: {output_path}")
    return output_path


def run_extended_amplification_report(
    model_names: list[str] | None = None,
    max_samples: int = 200,
    top_k: int = 10,
    fig_dir: str = "figures",
) -> str | None:
    model_names = model_names or ["gpt2-small", "pythia-160m"]
    scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("  Extended Amplification Cross-Model Comparison")
    print(f"  Grid: {scales}")
    print("=" * 70)

    model_failure_rates: dict[str, list[float]] = {}
    for model_name in model_names:
        print(f"\nEvaluating: {model_name}")
        model = load_model(model_name)
        pairs = load_counterfact(max_samples=max_samples, model=model)

        if not pairs:
            print(f"No valid pairs for {model_name}, skipping.")
            del model
            continue

        print(f"Loaded {len(pairs)} test samples.")
        print(f"Identifying top {top_k} inhibition heads...")
        mean_delta = compute_head_dla_batch(model, pairs, top_k=top_k)
        top_heads = select_top_heads(mean_delta, top_k=top_k)
        print(f"Top {top_k} heads: {top_heads}")

        print("Running dataset-level amplification sweep...")
        result = dataset_amplification_experiment(
            model,
            pairs=pairs,
            heads=top_heads,
            scales=scales,
            fig_dir=fig_dir,
            filename=f"{model_name}_amplification_failure_rate.png",
            verbose=True,
        )
        model_failure_rates[model_name] = result["failure_rates"]

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    figure_path = plot_failure_curves(model_failure_rates, scales, top_k, fig_dir)

    print("\nFinal Failure Rates (Scale=1.0 -> Scale=4.0):")
    for model_name in model_names:
        if model_name in model_failure_rates:
            rates = model_failure_rates[model_name]
            print(f"  {model_name:<12} : {rates[0] * 100:>5.1f}% -> {rates[-1] * 100:>5.1f}%")

    return figure_path


def main():
    args = parse_args()
    run_extended_amplification_report(
        model_names=args.models,
        max_samples=args.max_samples,
        top_k=args.top_k,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    main()
