"""
src/analysis/patching.py
========================
Activation patching utilities for negation experiments.

The core intervention replaces a chosen activation from the negated run
with the corresponding activation from the positive run at the final token
position. This follows the exploratory logic in ``transformerLenstest.py``
but packages it for reuse across scripts and models.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer


PATCH_TYPE_TO_HOOK = {
    "resid": "hook_resid_post",
    "mlp": "hook_mlp_out",
    "attn": "hook_attn_out",
}


def _dynamic_ylim(values: Iterable[float], floor: float | None = None, ceil: float | None = None, pad_ratio: float = 0.1):
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(vals.min())
        hi = float(vals.max())
        if np.isclose(lo, hi):
            pad = max(abs(lo) * pad_ratio, 0.1)
        else:
            pad = (hi - lo) * pad_ratio
        lo -= pad
        hi += pad

    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return lo, hi


def _patch_last_token(value: torch.Tensor, hook, source_cache) -> torch.Tensor:
    """Patch only the prediction position, where positive and negated runs differ."""
    value[:, -1, ...] = source_cache[hook.name][:, -1, ...]
    return value


@torch.no_grad()
def patched_prompt_metrics(
    model: HookedTransformer,
    positive_prompt: str,
    negated_prompt: str,
    target_token: str,
    patch_type: str,
    layer: int,
    top_k: int = 8,
) -> Dict:
    """
    Apply a single activation patch and return behavioural metrics.
    """
    if patch_type not in PATCH_TYPE_TO_HOOK:
        raise ValueError(f"Unknown patch_type '{patch_type}'. Expected one of {list(PATCH_TYPE_TO_HOOK)}.")

    target_id = model.to_single_token(target_token)
    hook_name = f"blocks.{layer}.{PATCH_TYPE_TO_HOOK[patch_type]}"
    _, pos_cache = model.run_with_cache(positive_prompt)
    patched_logits = model.run_with_hooks(
        negated_prompt,
        fwd_hooks=[(hook_name, partial(_patch_last_token, source_cache=pos_cache))],
    )
    last_logits = patched_logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, top_k)

    return {
        "target_logit": float(last_logits[target_id].item()),
        "target_prob": float(probs[target_id].item()),
        "target_rank": int((last_logits >= last_logits[target_id]).sum().item()),
        "top_predictions": [
            {"token": model.to_string(token_id), "prob": float(prob)}
            for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist())
        ],
    }


@torch.no_grad()
def activation_patching_scan(
    model: HookedTransformer,
    positive_prompt: str,
    negated_prompt: str,
    target_token: str,
    top_k: int = 8,
) -> Dict:
    """
    Scan residual, MLP, and attention patching across all layers.

    Returns the baseline negated-prompt behaviour, per-layer deltas for each
    patch type, and the single strongest patching intervention.
    """
    target_id = model.to_single_token(target_token)
    n_layers = model.cfg.n_layers

    _, pos_cache = model.run_with_cache(positive_prompt)
    baseline_logits = model(model.to_tokens(negated_prompt))
    last_logits = baseline_logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, top_k)

    baseline = {
        "target_logit": float(last_logits[target_id].item()),
        "target_prob": float(probs[target_id].item()),
        "target_rank": int((last_logits >= last_logits[target_id]).sum().item()),
        "top_predictions": [
            {"token": model.to_string(token_id), "prob": float(prob)}
            for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist())
        ],
    }

    results: Dict[str, Dict] = {}
    best_patch = None

    for patch_type, hook_suffix in PATCH_TYPE_TO_HOOK.items():
        deltas = []
        patched_logits = []
        patched_ranks = []
        patched_probs = []

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.{hook_suffix}"
            logits = model.run_with_hooks(
                negated_prompt,
                fwd_hooks=[(hook_name, partial(_patch_last_token, source_cache=pos_cache))],
            )
            layer_last_logits = logits[0, -1, :]
            layer_probs = torch.softmax(layer_last_logits, dim=-1)
            target_logit = float(layer_last_logits[target_id].item())
            target_prob = float(layer_probs[target_id].item())
            target_rank = int((layer_last_logits >= layer_last_logits[target_id]).sum().item())

            delta = target_logit - baseline["target_logit"]
            deltas.append(delta)
            patched_logits.append(target_logit)
            patched_probs.append(target_prob)
            patched_ranks.append(target_rank)

            if best_patch is None or delta > best_patch["delta"]:
                best_patch = {
                    "patch_type": patch_type,
                    "layer": layer,
                    "delta": delta,
                    "patched_logit": target_logit,
                    "patched_prob": target_prob,
                    "patched_rank": target_rank,
                }

        results[patch_type] = {
            "deltas": deltas,
            "patched_logits": patched_logits,
            "patched_probs": patched_probs,
            "patched_ranks": patched_ranks,
            "best_layer": int(np.argmax(deltas)),
            "best_delta": float(np.max(deltas)),
        }

    return {
        "baseline": baseline,
        "patches": results,
        "best_patch": best_patch,
    }


@torch.no_grad()
def dataset_activation_patching_experiment(
    model: HookedTransformer,
    pairs,
    max_samples: Optional[int] = None,
    fig_dir: str = "figures",
    filename: str = "activation_patching.png",
    title: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Average activation patching deltas across a dataset subset and save a figure.
    """
    selected_pairs = list(pairs[:max_samples] if max_samples is not None else pairs)
    n_layers = model.cfg.n_layers
    accum = {patch_type: [] for patch_type in PATCH_TYPE_TO_HOOK}
    analysed = 0

    if verbose:
        print(f"Running activation patching on {len(selected_pairs)} samples...")

    for pair in selected_pairs:
        try:
            scan = activation_patching_scan(
                model,
                pair.positive_prompt,
                pair.negated_prompt,
                pair.target_token,
                top_k=5,
            )
        except Exception:
            continue

        for patch_type in PATCH_TYPE_TO_HOOK:
            accum[patch_type].append(scan["patches"][patch_type]["deltas"])
        analysed += 1

    mean_deltas = {}
    if analysed == 0:
        for patch_type in PATCH_TYPE_TO_HOOK:
            mean_deltas[patch_type] = np.zeros(n_layers)
    else:
        for patch_type in PATCH_TYPE_TO_HOOK:
            mean_deltas[patch_type] = np.asarray(accum[patch_type], dtype=float).mean(axis=0)

    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, patch_type in zip(axes, PATCH_TYPE_TO_HOOK):
        values = mean_deltas[patch_type]
        ax.plot(range(n_layers), values, marker="o", linewidth=2)
        ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(f"{patch_type.upper()} patch")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Δ target logit")
        ax.set_ylim(*_dynamic_ylim(values))

    fig.suptitle(title or f"Activation patching summary ({analysed} samples)")
    plt.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    if verbose:
        print(f"Saved {path}")

    summary = {
        "n_samples": analysed,
        "mean_deltas": {k: v.tolist() for k, v in mean_deltas.items()},
        "best_layers": {
            k: {
                "layer": int(np.argmax(v)),
                "delta": float(np.max(v)),
            }
            for k, v in mean_deltas.items()
        },
        "figure_path": path,
    }
    return summary
