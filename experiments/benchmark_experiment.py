"""
Benchmark Experiment: Negation Failure Analysis

This script runs the full benchmarking pipeline:
1. Loads experiment configuration
2. Downloads the CounterFact dataset
3. Loads each specified model via TransformerLens
4. Runs the negation failure benchmark
5. Saves aggregate results to results/benchmark_results.json

Usage:
    python experiments/benchmark_experiment.py

Ensure you run this from the project root directory.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from src.dataset.load_dataset import load_counterfact
from src.models.load_models import load_model, clear_model_cache
from src.benchmark.run_benchmark import run_benchmark
from src.utils.io_utils import load_config, save_results


# --- Reproducibility ---
SEED = 68
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    """Run the full negation failure benchmark experiment."""

    # Paths
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "experiment_config.yaml"
    )
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "benchmark_results.json"
    )

    # 1. Load config
    print("=" * 60)
    print("  Negation Failure Benchmark Experiment")
    print("=" * 60)
    config = load_config(config_path)
    print(f"Config loaded: {config}")

    dataset_size = config.get("dataset_size", 100)
    model_names = config.get("models", ["gpt2-small"])
    device = config.get("device", "auto")

    # 2. Load dataset
    print(f"\nLoading CounterFact dataset (size={dataset_size})...")
    data = load_counterfact(dataset_size=dataset_size)
    print(f"Loaded {len(data)} samples.")

    # 3. Run benchmark for each model
    all_results = {}

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        # Load model
        model = load_model(model_name, device=device)

        # Run benchmark
        benchmark_output = run_benchmark(
            model=model,
            data=data,
            model_name=model_name,
        )

        # Store summary (exclude per-sample results for the summary file)
        all_results[model_name] = {
            "positive_accuracy": benchmark_output["positive_accuracy"],
            "negation_failure_rate": benchmark_output["negation_failure_rate"],
        }

        # Clear model from cache to free memory before loading next
        clear_model_cache()

    # 4. Save results
    print(f"\n{'=' * 60}")
    print("  Saving Results")
    print(f"{'=' * 60}")
    save_results(all_results, output_path)

    # 5. Print summary
    print(f"\n{'=' * 60}")
    print("  Final Summary")
    print(f"{'=' * 60}")
    for model_name, metrics in all_results.items():
        print(f"\n  {model_name}:")
        print(f"    Positive Accuracy:      {metrics['positive_accuracy']:.4f}")
        print(f"    Negation Failure Rate:  {metrics['negation_failure_rate']:.4f}")

    print(f"\nDone. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
