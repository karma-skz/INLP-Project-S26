"""
run_pipeline.py
================
Top-level entry point for the full Pink Elephants pipeline.

Stages
------
  1. Load model(s)
  2. Load CounterFact dataset
  3. Run DLA + SGR benchmark on each model
  4. Analyse SGR distribution and generate figures
  5. Per-head decomposition to identify inhibition heads
  6. Artificial amplification experiment
  7. Statistical analysis and final report

Usage
-----
    # Minimal run — GPT-2, 200 samples
    python run_pipeline.py

    # Full run — both models, all samples
    python run_pipeline.py --models gpt2-small pythia-160m --max_samples -1

    # Skip heavy stages
    python run_pipeline.py --skip_patching --skip_amplification
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from src.dataset  import load_counterfact
from src.models   import load_model
from src.benchmark import run_benchmark, analyse_sgr_distribution
from src.analysis  import per_head_dla, top_inhibition_heads, compute_head_dla_batch
from src.analysis  import amplification_sweep, dataset_amplification_experiment
from src.metrics   import summary_stats, sgr_vs_failure_correlation, compare_models, negation_failure_rate

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(67)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pink Elephants — full pipeline")
    p.add_argument("--models", nargs="+",
                   default=["gpt2-small"],
                   choices=["gpt2-small", "gpt2", "pythia-160m", "pythia"],
                   help="Models to benchmark (default: gpt2-small)")
    p.add_argument("--max_samples", type=int, default=200,
                   help="Max CounterFact samples per model (-1 = all, default: 200)")
    p.add_argument("--results_dir", default="results",
                   help="Directory for CSV outputs (default: results/)")
    p.add_argument("--fig_dir", default="figures",
                   help="Directory for figures (default: figures/)")
    p.add_argument("--top_k_heads", type=int, default=10,
                   help="Number of top inhibition heads to select for amplification")
    p.add_argument("--amp_scales", nargs="+", type=float,
                   default=[0.5, 1.0, 2.0, 3.0, 4.0],
                   help="Amplification scales to sweep (default: 0.5 1 2 3 4)")
    p.add_argument("--skip_per_head", action="store_true",
                   help="Skip per-head decomposition")
    p.add_argument("--skip_amplification", action="store_true",
                   help="Skip amplification experiment")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_shortname(name: str) -> str:
    """Normalise 'gpt2' → 'gpt2-small', etc."""
    mapping = {"gpt2": "gpt2-small", "pythia": "pythia-160m"}
    return mapping.get(name, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir,     exist_ok=True)

    model_names = [_model_shortname(m) for m in args.models]
    max_s       = None if args.max_samples < 0 else args.max_samples

    all_dfs = []

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1–3: Benchmark each model
    # ══════════════════════════════════════════════════════════════════════════
    for model_name in model_names:
        print(f"\n{'#'*70}")
        print(f"# Model: {model_name}")
        print(f"{'#'*70}\n")

        model = load_model(model_name)

        # Load dataset (validate single-token targets against this model)
        pairs = load_counterfact(max_samples=max_s, model=model)

        # Run benchmark
        csv_path = os.path.join(args.results_dir, f"{model_name}_benchmark.csv")
        df = run_benchmark(
            model,
            pairs,
            model_name=model_name,
            output_csv=csv_path,
        )
        all_dfs.append(df)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: SGR distribution analysis
    # ══════════════════════════════════════════════════════════════════════════
    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'#'*70}")
    print("# Stage 4: SGR Distribution Analysis")
    print(f"{'#'*70}\n")

    analyse_sgr_distribution(combined_df, fig_dir=args.fig_dir)

    # Combined CSV
    combined_csv = os.path.join(args.results_dir, "all_models_benchmark.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"Combined results → {combined_csv}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5: Per-head decomposition (first model, subsample for speed)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_per_head:
        primary_model_name = model_names[0]
        print(f"\n{'#'*70}")
        print(f"# Stage 5: Per-Head Decomposition ({primary_model_name})")
        print(f"{'#'*70}\n")

        p_model = load_model(primary_model_name)
        p_pairs = load_counterfact(max_samples=min(max_s or 200, 200), model=p_model)

        # Compute mean delta across dataset
        mean_delta = compute_head_dla_batch(
            p_model, p_pairs, top_k=args.top_k_heads
        )

        # Identify top inhibition heads
        n_heads = p_model.cfg.n_heads
        flat    = mean_delta.flatten().argsort()[::-1][:args.top_k_heads]
        top_heads = [
            (int(idx // n_heads), int(idx % n_heads))
            for idx in flat
        ]
        print(f"\nTop-{args.top_k_heads} inhibition heads: {top_heads}")

        # Visualise on the canonical example
        from src.analysis.per_head import plot_head_dla_heatmap
        h_pos, h_neg = per_head_dla(
            p_model,
            "The capital of France is",
            "The capital of France is not",
            " Paris",
        )
        plot_head_dla_heatmap(h_pos, h_neg,
                              target_token=" Paris",
                              fig_dir=args.fig_dir)

        # ══════════════════════════════════════════════════════════════════════
        # Stage 6: Amplification experiment
        # ══════════════════════════════════════════════════════════════════════
        if not args.skip_amplification:
            print(f"\n{'#'*70}")
            print("# Stage 6: Artificial Amplification")
            print(f"{'#'*70}\n")

            # Single-prompt sweep
            amplification_sweep(
                p_model,
                positive_prompt="The capital of France is",
                negated_prompt="The capital of France is not",
                target_token=" Paris",
                heads=top_heads,
                scales=args.amp_scales,
                fig_dir=args.fig_dir,
            )

            # Dataset-level failure rate vs scale
            dataset_amplification_experiment(
                p_model,
                pairs=p_pairs,
                heads=top_heads,
                scales=args.amp_scales,
                fig_dir=args.fig_dir,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 7: Statistical analysis
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print("# Stage 7: Statistical Analysis")
    print(f"{'#'*70}\n")

    summary_stats(combined_df)

    corr_df = sgr_vs_failure_correlation(combined_df)
    print("\nSGR ↔ Failure Correlations:")
    print(corr_df.to_string(index=False))

    nf_df = negation_failure_rate(combined_df)
    print("\nNegation Failure Rates (with 95% CI):")
    print(nf_df.to_string(index=False))

    # Cross-model comparison (if >1 model)
    if len(model_names) > 1:
        compare_models(combined_df, model_names[0], model_names[1])

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"  Results  → {args.results_dir}/")
    print(f"  Figures  → {args.fig_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
