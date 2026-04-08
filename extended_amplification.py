"""
extended_amplification.py
=========================
Finer-scale artificial amplification testing across both GPT-2 and Pythia.
Measures whether amplifying the top inhibition heads suppresses the Pink
Elephant hallucination more effectively in Pythia than GPT-2 (as hypothesised).

Usage
-----
  conda run -n inlp-project python extended_amplification.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Monkey patch for transformer_lens Pythia/GPTNeoX loading bug 
try:
    from transformers import GPTNeoXConfig
    if not hasattr(GPTNeoXConfig, "rotary_pct"):
        GPTNeoXConfig.rotary_pct = 0.25
except ImportError:
    pass

# We intentionally avoid importing torch globally if possible to save memory

from src.dataset import load_counterfact
from src.models import load_model
from src.analysis import compute_head_dla_batch
from src.analysis.amplification import dataset_amplification_experiment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=200,
                   help="Number of dataset samples to run per model.")
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of top inhibition heads to amplify per model.")
    p.add_argument("--fig_dir", type=str, default="figures")
    return p.parse_args()


def get_top_inhibition_heads(model, pairs, top_k: int):
    """Compute and return the top_k inhibition heads for a given model."""
    mean_delta = compute_head_dla_batch(model, pairs, top_k=top_k)
    n_heads = model.cfg.n_heads
    flat = mean_delta.flatten().argsort()[::-1][:top_k]
    return [(int(idx // n_heads), int(idx % n_heads)) for idx in flat]


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    
    # ── Configuration ──
    models_to_test = ["gpt2-small", "pythia-160m"]
    scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    
    # We will collect the failure rates per model
    model_failure_rates = {}

    print("=" * 70)
    print("  Extended Amplification Cross-Model Comparison")
    print(f"  Grid: {scales}")
    print("=" * 70)

    for model_name in models_to_test:
        print(f"\nEvaluating: {model_name}")
        
        # 1. Load Model
        # (Loaded sequentially and deleted afterwards to bound memory)
        model = load_model(model_name)
        
        # 2. Load Dataset Pairs
        # (Must re-load per model to filter out multi-token targets properly)
        pairs = load_counterfact(max_samples=args.max_samples, model=model)
        
        if len(pairs) == 0:
            print(f"No valid pairs for {model_name}, skipping.")
            del model
            continue
            
        print(f"Loaded {len(pairs)} test samples.")

        # 3. Compute Inhibition Heads specific to this model architecture
        print(f"Identifying top {args.top_k} inhibition heads...")
        top_heads = get_top_inhibition_heads(model, pairs, top_k=args.top_k)
        print(f"Top {args.top_k} heads: {top_heads}")
        
        # 4. Sweep Amplification
        print("Running dataset-level amplification sweep...")
        # Since the function natively plots and saves to figures/amplification_failure_rate.png, 
        # it will overwrite each loop. That's fine, we primarily need the returned list.
        failure_rates = dataset_amplification_experiment(
            model,
            pairs=pairs,
            heads=top_heads,
            scales=scales,
            fig_dir=args.fig_dir,
            verbose=True
        )
        
        model_failure_rates[model_name] = failure_rates
        
        # Explicit memory cleanup
        import gc
        del model
        gc.collect()

    print("\n" + "=" * 70)
    print("  Generating Cross-Model Comparative Figure")
    print("=" * 70)

    # ── Generate Dual Comparative Plot ──
    if not model_failure_rates:
        print("No valid results computed.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    colors = {"gpt2-small": "steelblue", "pythia-160m": "mediumorchid"}
    markers = {"gpt2-small": "o", "pythia-160m": "s"}
    
    for m_name in models_to_test:
        if m_name in model_failure_rates:
            frs = [fr * 100 for fr in model_failure_rates[m_name]] # converted to %
            ax.plot(scales, frs, marker=markers[m_name], color=colors[m_name], 
                    linewidth=2.5, markersize=7, label=f"{m_name} (Top {args.top_k} heads)")

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.2, label="Baseline (scale=1.0)")
    
    ax.set_xlabel(r"Inhibition Head Amplification Scale ($\alpha$)", fontsize=11)
    ax.set_ylabel("Negation Failure Rate (%)", fontsize=11)
    ax.set_title("Cross-Model Amplification: Pythia vs GPT-2\nDoes amplifying logic help larger models more?", 
                 fontsize=12, fontweight="bold")
    
    ax.set_ylim(0, max([max(rates)*100 for rates in model_failure_rates.values()]) * 1.2)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    ax.set_facecolor("#f8f9fa")
    
    plt.tight_layout()
    final_path = os.path.join(args.fig_dir, "extended_amplification.png")
    plt.savefig(final_path, dpi=160)
    plt.close()
    
    print(f"Success! Final comparative graph saved to: {final_path}")
    print("\nFinal Failure Rates (Scale=1.0 -> Scale=4.0):")
    for m_name in models_to_test:
        if m_name in model_failure_rates:
            frs = model_failure_rates[m_name]
            print(f"  {m_name:<12} : {frs[0]*100:>5.1f}% -> {frs[-1]*100:>5.1f}%")

if __name__ == "__main__":
    main()
