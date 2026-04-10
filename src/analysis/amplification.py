"""
src/analysis/amplification.py
===============================
Artificial amplification experiment — multiply inhibition head outputs by a
scalar factor and measure whether the negation failure is corrected.

Conceptually: if a head is an "inhibition head" but its output is too weak
to overcome the FFN retrieval signal, scaling it up should reduce the target
logit on negated prompts without affecting the positive prompt.

Usage
-----
    from src.models import load_model
    from src.analysis import amplify_heads, amplification_sweep

    model = load_model("gpt2-small")

    # Amplify a specific set of heads by 3× on the negated prompt
    result = amplify_heads(
        model,
        prompt="The capital of France is not",
        target_token=" Paris",
        heads=[(9, 6), (10, 0)],   # (layer, head) pairs
        scale=3.0,
    )
    print(result)  # {"baseline_logit": ..., "amplified_logit": ..., "delta": ...}

    # Sweep over scale values to produce a figure
    amplification_sweep(
        model, prompt, " Paris", heads, scales=[0.5, 1, 2, 3, 4, 5]
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------

def _make_head_scale_hook(head_idx: int, scale: float):
    """
    Return a hook function that scales the output of a specific attention head.

    TransformerLens stores the per-head attention result in
    ``blocks.{layer}.attn.hook_result`` with shape (batch, seq, n_heads, d_head).
    We scale only the chosen head index.
    """
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        # value shape: (batch, seq, n_heads, d_head)
        value[:, :, head_idx, :] = value[:, :, head_idx, :] * scale
        return value
    return hook_fn


# ---------------------------------------------------------------------------
# Single amplification run
# ---------------------------------------------------------------------------

@torch.no_grad()
def amplify_heads(
    model: HookedTransformer,
    prompt: str,
    target_token: str,
    heads: List[Tuple[int, int]],
    scale: float,
) -> Dict[str, float]:
    """
    Run *prompt* with specified attention heads scaled by *scale* and measure
    the change in target logit.

    Parameters
    ----------
    model : HookedTransformer
    prompt : str
        Typically the *negated* prompt (where we want suppression to improve).
    target_token : str
        Single token, e.g. ``" Paris"``.
    heads : list of (layer, head) tuples
        Attention heads to amplify.
    scale : float
        Multiplicative factor (1.0 = no change, 2.0 = double, etc.)

    Returns
    -------
    dict with keys:
      ``baseline_logit``   — original target logit (no amplification)
      ``amplified_logit``  — target logit with amplified heads
      ``delta``            — amplified - baseline
      ``baseline_rank``    — token rank before amplification
      ``amplified_rank``   — token rank after amplification
    """
    target_id = model.to_single_token(target_token)

    # Baseline (no hooks)
    baseline_logits = model(model.to_tokens(prompt))
    baseline_logit  = baseline_logits[0, -1, target_id].item()
    baseline_rank   = int((baseline_logits[0, -1, :] >= baseline_logits[0, -1, target_id]).sum().item())

    # Build hooks
    fwd_hooks = []
    for layer, head in heads:
        hook_name = f"blocks.{layer}.attn.hook_result"
        fwd_hooks.append((hook_name, _make_head_scale_hook(head, scale)))

    # Amplified run
    amp_logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
    amp_logit  = amp_logits[0, -1, target_id].item()
    amp_rank   = int((amp_logits[0, -1, :] >= amp_logits[0, -1, target_id]).sum().item())

    return {
        "baseline_logit":  baseline_logit,
        "amplified_logit": amp_logit,
        "delta":           amp_logit - baseline_logit,
        "baseline_rank":   baseline_rank,
        "amplified_rank":  amp_rank,
    }


# ---------------------------------------------------------------------------
# Sweep over scale values
# ---------------------------------------------------------------------------

@torch.no_grad()
def amplification_sweep(
    model: HookedTransformer,
    positive_prompt: str,
    negated_prompt: str,
    target_token: str,
    heads: List[Tuple[int, int]],
    scales: Optional[List[float]] = None,
    fig_dir: str = "figures",
    filename: str = "amplification_sweep.png",
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Sweep amplification scale 0.25 → 8× and plot how the target logit changes
    on both the positive and negated prompts.

    The ideal result:
      • Negated logit DECREASES  (amplifying inhibition suppresses " Paris")
      • Positive logit UNCHANGED or slightly affected

    Parameters
    ----------
    model : HookedTransformer
    positive_prompt, negated_prompt : str
    target_token : str
    heads : list of (layer, head) — the inhibition heads to amplify
    scales : list of float, optional  (default: [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8])
    fig_dir : str
    filename : str
    verbose : bool

    Returns
    -------
    dict with keys ``scales``, ``pos_logits``, ``neg_logits``
    """
    if scales is None:
        scales = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

    pos_logits = []
    neg_logits = []

    for scale in scales:
        pos_res = amplify_heads(model, positive_prompt, target_token, heads, scale)
        neg_res = amplify_heads(model, negated_prompt,  target_token, heads, scale)
        pos_logits.append(pos_res["amplified_logit"])
        neg_logits.append(neg_res["amplified_logit"])
        if verbose:
            print(f"  scale={scale:.2f}  pos_logit={pos_logits[-1]:+.3f}  "
                  f"neg_logit={neg_logits[-1]:+.3f}  "
                  f"gap={pos_logits[-1]-neg_logits[-1]:+.3f}")

    gap = [p - n for p, n in zip(pos_logits, neg_logits)]
    del fig_dir, filename

    best_gap_idx = int(np.argmax(gap)) if gap else 0
    return {
        "scales": scales,
        "pos_logits": pos_logits,
        "neg_logits": neg_logits,
        "gap": gap,
        "best_gap_scale": scales[best_gap_idx] if scales else np.nan,
        "best_gap": gap[best_gap_idx] if gap else np.nan,
    }


# ---------------------------------------------------------------------------
# Dataset-level amplification: measure failure rate at different scales
# ---------------------------------------------------------------------------

@torch.no_grad()
def dataset_amplification_experiment(
    model: HookedTransformer,
    pairs,                   # list of PromptPair
    heads: List[Tuple[int, int]],
    scales: Optional[List[float]] = None,
    fig_dir: str = "figures",
    filename: str = "amplification_failure_rate.png",
    verbose: bool = True,
) -> Dict[str, object]:
    """
    For each scale, compute the negation failure rate over *pairs* and plot
    how it decreases as inhibition heads are amplified.

    Parameters
    ----------
    model : HookedTransformer
    pairs : list of PromptPair
    heads : list of (layer, head)
    scales : list of float
    fig_dir : str
    verbose : bool
    """
    from tqdm import tqdm

    if scales is None:
        scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    prepared_pairs: list[tuple[object, int, int]] = []
    for pair in pairs:
        try:
            target_id = model.to_single_token(pair.target_token)
        except Exception:
            continue

        pos_logits = model(model.to_tokens(pair.positive_prompt))
        pos_rank = int(
            (pos_logits[0, -1, :] >= pos_logits[0, -1, target_id]).sum().item()
        )
        prepared_pairs.append((pair, target_id, pos_rank))

    failure_rates = []

    for scale in scales:
        failures = 0
        total    = 0

        for pair, target_id, pos_rank in tqdm(prepared_pairs, desc=f"Amp scale={scale:.1f}", leave=False):
            fwd_hooks = [
                (f"blocks.{layer}.attn.hook_result",
                 _make_head_scale_hook(head, scale))
                for layer, head in heads
            ]

            amp_logits = model.run_with_hooks(pair.negated_prompt, fwd_hooks=fwd_hooks)
            amp_rank   = int(
                (amp_logits[0, -1, :] >= amp_logits[0, -1, target_id]).sum().item()
            )

            if amp_rank < pos_rank:
                failures += 1
            total += 1

        rate = failures / total if total > 0 else float("nan")
        failure_rates.append(rate)
        if verbose:
            print(f"  scale={scale:.2f}  failure_rate={rate:.1%}  ({failures}/{total})")

    del fig_dir, filename

    baseline_idx = scales.index(1.0) if 1.0 in scales else None
    baseline_rate = failure_rates[baseline_idx] if baseline_idx is not None else np.nan
    best_idx = int(np.nanargmin(failure_rates)) if failure_rates else 0
    best_rate = failure_rates[best_idx] if failure_rates else np.nan

    return {
        "scales": scales,
        "failure_rates": failure_rates,
        "baseline_rate": baseline_rate,
        "best_rate": best_rate,
        "best_scale": scales[best_idx] if scales else np.nan,
        "absolute_improvement": (baseline_rate - best_rate) if np.isfinite(baseline_rate) and np.isfinite(best_rate) else np.nan,
        "n_pairs": len(prepared_pairs),
    }
