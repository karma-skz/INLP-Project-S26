"""
activation_patching.py
======================
Main dataset-scale Activation Patching pipeline.
Sweeps across the dataset to find naturally failing (hallucinating) prompt pairs,
then sequentially patches the Residual Stream, MLPs, and Attention Outputs layer-by-layer
using cleanly computed positive logic to measure the "Rescue Rate".

Usage
-----
    conda run -n inlp-project python activation_patching.py
"""

import argparse
import os
import gc
from functools import partial

import torch
import numpy as np
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

from src.dataset import load_counterfact
from src.models import load_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gpt2-small", "pythia-160m"],
                   help="List of HF models to evaluate.")
    p.add_argument("--max_samples", type=int, default=50,
                   help="Number of dataset samples to run per model (for testing).")
    p.add_argument("--fig_dir", type=str, default="figures")
    return p.parse_args()


def patch_hook_fn(value, hook, pos_cache):
    """
    Patches the specified stream at the target layer with the clean activation.
    Only patches sequence position -1 (the prediction position).
    """
    pos_act = pos_cache[hook.name]
    value[:, -1, :] = pos_act[:, -1, :]
    return value


def get_rank(logits, target_id):
    """Calculate the rank of the target token (1 = top prediction)."""
    last_logits = logits[0, -1, :]
    return (last_logits >= last_logits[target_id]).sum().item()


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    
    models_to_test = args.models
    
    # Store rates across models
    all_rescue_rates = {}

    print("=" * 70)
    print("  Main-Pipeline Activation Patching Evaluation")
    print("=" * 70)

    for model_name in models_to_test:
        print(f"\nEvaluating: {model_name}")
        model = load_model(model_name)
        n_layers = model.cfg.n_layers
        
        pairs = load_counterfact(max_samples=args.max_samples, model=model)
        if not pairs:
            del model
            continue
            
        print(f"Loaded {len(pairs)} test samples. Identifying failure cases...")
        
        failure_pairs = []
        pos_ranks_for_failures = []
        
        # 1. Identify native failure cases
        with torch.no_grad():
            for pair in pairs:
                target_id = model.to_single_token(pair.target_token)
                
                pos_tokens = model.to_tokens(pair.positive_prompt)
                neg_tokens = model.to_tokens(pair.negated_prompt)
                
                pos_logits = model(pos_tokens)
                neg_logits = model(neg_tokens)
                
                pos_rank = get_rank(pos_logits, target_id)
                neg_rank = get_rank(neg_logits, target_id)
                
                # If negative prompt ranks the target closer to top 1 than positive prompt,
                # the model failed to suppress the target (hallucinated).
                if neg_rank < pos_rank:
                    failure_pairs.append((pair, target_id, pos_tokens, neg_tokens))
                    pos_ranks_for_failures.append(pos_rank)
                    
        total_failures = len(failure_pairs)
        print(f"Found {total_failures} natural failure cases ({(total_failures/len(pairs))*100:.1f}% failure rate).")
        
        if total_failures == 0:
            print("No failures to rescue. Skipping.")
            del model
            continue
            
        # 2. Rescue Patching Loops
        # We track how many failures are cured at each layer for 3 components.
        rescues_resid = np.zeros(n_layers)
        rescues_mlp = np.zeros(n_layers)
        rescues_attn = np.zeros(n_layers)
        
        import tqdm
        print("Commencing Patching Iterations:")
        with torch.no_grad():
            for i, (pair, target_id, pos_tokens, neg_tokens) in enumerate(tqdm.tqdm(failure_pairs, desc=model_name)):
                # Get clean cached activations
                _, pos_cache = model.run_with_cache(pos_tokens)
                
                baseline_safe_rank = pos_ranks_for_failures[i]
                
                for layer in range(n_layers):
                    # Patch Residual
                    l_resid = model.run_with_hooks(
                        neg_tokens, 
                        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", partial(patch_hook_fn, pos_cache=pos_cache))]
                    )
                    if get_rank(l_resid, target_id) >= baseline_safe_rank:
                        rescues_resid[layer] += 1
                        
                    # Patch MLP
                    l_mlp = model.run_with_hooks(
                        neg_tokens, 
                        fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", partial(patch_hook_fn, pos_cache=pos_cache))]
                    )
                    if get_rank(l_mlp, target_id) >= baseline_safe_rank:
                        rescues_mlp[layer] += 1
                        
                    # Patch Attention
                    l_attn = model.run_with_hooks(
                        neg_tokens, 
                        fwd_hooks=[(f"blocks.{layer}.hook_attn_out", partial(patch_hook_fn, pos_cache=pos_cache))]
                    )
                    if get_rank(l_attn, target_id) >= baseline_safe_rank:
                        rescues_attn[layer] += 1

        all_rescue_rates[model_name] = {
            "n_layers": n_layers,
            "resid": (rescues_resid / total_failures) * 100,
            "mlp": (rescues_mlp / total_failures) * 100,
            "attn": (rescues_attn / total_failures) * 100,
        }
        
        # Cleanup
        del model
        gc.collect()

    # 3. Plotting Results
    print("\n" + "=" * 70)
    print("  Generating Rescue Rate Comparative Figure")
    print("=" * 70)
    
    if not all_rescue_rates:
        print("No plot to generate.")
        return

    # Create subplots for however many models we tested
    n_models = len(all_rescue_rates)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
        
    for idx, (m_name, rates) in enumerate(all_rescue_rates.items()):
        ax = axes[idx]
        layers = range(rates["n_layers"])
        
        ax.plot(layers, rates["resid"], marker="o", color="mediumpurple", linewidth=2, label="Residual Stream")
        ax.plot(layers, rates["mlp"], marker="s", color="salmon", linewidth=2, label="MLP Output")
        ax.plot(layers, rates["attn"], marker="^", color="steelblue", linewidth=2, label="Attn Output")
        
        ax.set_title(f"{m_name} (n_layers={rates['n_layers']})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Patched Layer", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Rescue Rate (% of native failures fixed)", fontsize=11)
        ax.legend()
        
    plt.suptitle("Main-Pipeline Activation Patching\nEffectiveness of Causal Interventions per Layer", 
                 fontsize=14, fontweight="bold", y=1.05)
    
    plt.tight_layout()
    out_path = os.path.join(args.fig_dir, "activation_patching_rescue_rate.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Success! Final figure saved to: {out_path}")

if __name__ == "__main__":
    main()
