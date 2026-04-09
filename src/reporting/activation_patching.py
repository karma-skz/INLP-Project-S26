from __future__ import annotations

import argparse
import gc
import os
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import load_counterfact
from src.models import load_model
from src.utils import dynamic_axis_limits


matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset-scale activation patching rescue-rate report")
    parser.add_argument("--models", nargs="+", default=["gpt2-small", "pythia-160m"], help="Models to evaluate")
    parser.add_argument("--max_samples", type=int, default=50, help="Number of dataset samples to test per model")
    parser.add_argument("--fig_dir", default="figures", help="Directory to save figures into")
    return parser.parse_args()


def patch_last_token(value, hook, source_cache):
    source_activation = source_cache[hook.name]
    value[:, -1, :] = source_activation[:, -1, :]
    return value


def token_rank(logits: torch.Tensor, target_id: int) -> int:
    last_logits = logits[0, -1, :]
    return int((last_logits >= last_logits[target_id]).sum().item())


@torch.no_grad()
def find_failure_pairs(model, pairs):
    failures = []
    positive_ranks = []

    for pair in pairs:
        target_id = model.to_single_token(pair.target_token)
        pos_tokens = model.to_tokens(pair.positive_prompt)
        neg_tokens = model.to_tokens(pair.negated_prompt)

        pos_logits = model(pos_tokens)
        neg_logits = model(neg_tokens)
        pos_rank = token_rank(pos_logits, target_id)
        neg_rank = token_rank(neg_logits, target_id)

        if neg_rank < pos_rank:
            failures.append((pair, target_id, pos_tokens, neg_tokens))
            positive_ranks.append(pos_rank)

    return failures, positive_ranks


@torch.no_grad()
def rescue_rates_for_model(model, pairs) -> dict | None:
    n_layers = model.cfg.n_layers
    failure_pairs, baseline_safe_ranks = find_failure_pairs(model, pairs)

    total_failures = len(failure_pairs)
    if total_failures == 0:
        return None

    rescues_resid = np.zeros(n_layers)
    rescues_mlp = np.zeros(n_layers)
    rescues_attn = np.zeros(n_layers)

    import tqdm

    for index, (_, target_id, pos_tokens, neg_tokens) in enumerate(tqdm.tqdm(failure_pairs, desc=model.cfg.model_name)):
        _, pos_cache = model.run_with_cache(pos_tokens)
        baseline_safe_rank = baseline_safe_ranks[index]

        for layer in range(n_layers):
            resid_logits = model.run_with_hooks(
                neg_tokens,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", partial(patch_last_token, source_cache=pos_cache))],
            )
            if token_rank(resid_logits, target_id) >= baseline_safe_rank:
                rescues_resid[layer] += 1

            mlp_logits = model.run_with_hooks(
                neg_tokens,
                fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", partial(patch_last_token, source_cache=pos_cache))],
            )
            if token_rank(mlp_logits, target_id) >= baseline_safe_rank:
                rescues_mlp[layer] += 1

            attn_logits = model.run_with_hooks(
                neg_tokens,
                fwd_hooks=[(f"blocks.{layer}.hook_attn_out", partial(patch_last_token, source_cache=pos_cache))],
            )
            if token_rank(attn_logits, target_id) >= baseline_safe_rank:
                rescues_attn[layer] += 1

    return {
        "n_layers": n_layers,
        "total_failures": total_failures,
        "resid": (rescues_resid / total_failures) * 100,
        "mlp": (rescues_mlp / total_failures) * 100,
        "attn": (rescues_attn / total_failures) * 100,
    }


def plot_rescue_rates(all_rates: dict[str, dict], fig_dir: str) -> str | None:
    if not all_rates:
        print("No plot to generate.")
        return None

    os.makedirs(fig_dir, exist_ok=True)
    n_models = len(all_rates)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    all_values = []
    for rates in all_rates.values():
        all_values.extend(rates["resid"])
        all_values.extend(rates["mlp"])
        all_values.extend(rates["attn"])

    for axis, (model_name, rates) in zip(axes, all_rates.items()):
        layers = range(rates["n_layers"])
        axis.plot(layers, rates["resid"], marker="o", color="mediumpurple", linewidth=2, label="Residual stream")
        axis.plot(layers, rates["mlp"], marker="s", color="salmon", linewidth=2, label="MLP output")
        axis.plot(layers, rates["attn"], marker="^", color="steelblue", linewidth=2, label="Attention output")
        axis.set_title(f"{model_name} (n_layers={rates['n_layers']})", fontsize=12, fontweight="bold")
        axis.set_xlabel("Patched layer", fontsize=11)
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.set_ylim(*dynamic_axis_limits(all_values, floor=0.0, ceil=100.0))
        if axis is axes[0]:
            axis.set_ylabel("Rescue rate (% of failures fixed)", fontsize=11)
        axis.legend()

    plt.suptitle("Activation Patching Rescue Rate", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    output_path = os.path.join(fig_dir, "activation_patching_rescue_rate.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Success! Final figure saved to: {output_path}")
    return output_path


def run_activation_patching_report(
    model_names: list[str] | None = None,
    max_samples: int = 50,
    fig_dir: str = "figures",
) -> str | None:
    model_names = model_names or ["gpt2-small", "pythia-160m"]
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("  Main-Pipeline Activation Patching Evaluation")
    print("=" * 70)

    all_rescue_rates = {}
    for model_name in model_names:
        print(f"\nEvaluating: {model_name}")
        model = load_model(model_name)
        pairs = load_counterfact(max_samples=max_samples, model=model)
        if not pairs:
            del model
            continue

        print(f"Loaded {len(pairs)} test samples. Identifying failure cases...")
        rates = rescue_rates_for_model(model, pairs)
        if rates is None:
            print("No failures to rescue. Skipping.")
            del model
            gc.collect()
            continue

        print(
            f"Found {rates['total_failures']} natural failure cases "
            f"({(rates['total_failures'] / len(pairs)) * 100:.1f}% failure rate)."
        )
        all_rescue_rates[model_name] = rates

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return plot_rescue_rates(all_rescue_rates, fig_dir)


def main():
    args = parse_args()
    run_activation_patching_report(model_names=args.models, max_samples=args.max_samples, fig_dir=args.fig_dir)


if __name__ == "__main__":
    main()
