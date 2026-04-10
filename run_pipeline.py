"""
Lightweight pipeline entrypoint for running the benchmark and intervention
summaries without generating graphs.
"""

from __future__ import annotations

import argparse
import gc
import os

import pandas as pd
import torch

from src.analysis import compute_head_dla_batch, dataset_activation_patching_experiment, dataset_amplification_experiment, select_top_heads
from src.benchmark import analyse_sgr_distribution, run_benchmark
from src.dataset import load_counterfact
from src.models import CANONICAL_MODEL_NAMES, MODEL_SHORTNAMES, load_model
from src.utils import benchmark_csv_path


torch.manual_seed(67)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the main negation benchmark without graphs")
    parser.add_argument("--models", nargs="+", default=CANONICAL_MODEL_NAMES, choices=list(MODEL_SHORTNAMES.keys()), help="Models to benchmark")
    parser.add_argument("--negator_suffix", default=" not", help="Suffix appended to build negated prompts")
    parser.add_argument("--max_samples", type=int, default=200, help="Max CounterFact samples per model (-1 = all)")
    parser.add_argument("--results_dir", default="results", help="Directory for CSV outputs")
    parser.add_argument("--analysis_samples", type=int, default=100, help="Samples used for head selection and amplification")
    parser.add_argument("--patching_samples", type=int, default=20, help="Samples used for dataset patching summaries")
    parser.add_argument("--top_k_heads", type=int, default=10, help="Number of top inhibition heads to summarize")
    parser.add_argument("--amp_scales", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0], help="Amplification scales")
    parser.add_argument("--skip_interventions", action="store_true", help="Skip head-selection, amplification, and patching summaries")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    model_names = [MODEL_SHORTNAMES.get(model_name, model_name) for model_name in args.models]
    max_samples = None if args.max_samples < 0 else args.max_samples

    all_dfs: list[pd.DataFrame] = []

    for model_name in model_names:
        print("\n" + "#" * 70)
        print(f"# Model: {model_name}")
        print("#" * 70 + "\n")

        model = load_model(model_name)
        pairs = load_counterfact(max_samples=max_samples, model=model, negator_suffix=args.negator_suffix)

        csv_path = str(benchmark_csv_path(args.results_dir, model_name, args.negator_suffix))
        df = run_benchmark(model, pairs, model_name=model_name, output_csv=csv_path)
        df["negator"] = args.negator_suffix
        df.to_csv(csv_path, index=False)
        all_dfs.append(df)

        if not args.skip_interventions:
            analysis_pairs = pairs[: min(len(pairs), args.analysis_samples)]
            patch_pairs = analysis_pairs[: min(len(analysis_pairs), args.patching_samples)]

            mean_delta = compute_head_dla_batch(model, analysis_pairs, top_k=args.top_k_heads)
            top_heads = select_top_heads(mean_delta, top_k=args.top_k_heads)
            print(f"Top heads for {model_name}: {top_heads[:5]}")

            amp_summary = dataset_amplification_experiment(
                model,
                pairs=analysis_pairs,
                heads=top_heads,
                scales=args.amp_scales,
                verbose=True,
            )
            patch_summary = dataset_activation_patching_experiment(
                model,
                patch_pairs,
                max_samples=None,
                verbose=True,
            )

            print(
                f"Amplification summary for {model_name}: baseline={amp_summary['baseline_rate']:.1%}, "
                f"best={amp_summary['best_rate']:.1%} @ scale={amp_summary['best_scale']:.2f}"
            )
            best_patch = patch_summary["best_overall"]
            if best_patch is not None:
                print(
                    f"Patching summary for {model_name}: best={best_patch['patch_type']} "
                    f"L{best_patch['layer']} (Δ {best_patch['delta']:+.3f})"
                )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(args.results_dir, "all_models_benchmark.csv")
    combined_df.to_csv(combined_csv, index=False)

    sgr_summary = analyse_sgr_distribution(combined_df, verbose=False)
    print("\nBenchmark summary:")
    print(
        sgr_summary["benchmark_summary"][
            ["model_name", "n_samples", "n_failures", "failure_rate", "median_rank_shift", "median_sgr"]
        ].to_string(index=False)
    )
    print("\nSGR edge cases:")
    print(sgr_summary["edge_cases"].to_string(index=False))

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Results -> {args.results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
