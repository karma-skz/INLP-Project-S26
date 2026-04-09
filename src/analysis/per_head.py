"""
src/analysis/per_head.py
=========================
Per-attention-head DLA decomposition to identify specific "Inhibition Heads"
(following Hanna et al., 2023).

For each attention head (layer, head) we compute how much that head's
output contributes to the final logit for the target token — separately
for the positive and negated prompt.

Inhibition heads are those whose DLA is:
  • large and POSITIVE on the positive prompt  (they promote the target)
  • large and NEGATIVE on the negated prompt   (they try to suppress it)

Their difference (DLA_pos - DLA_neg) quantifies "how much this head
reacts to the negation signal".

Usage
-----
    from src.models import load_model
    from src.analysis import per_head_dla, top_inhibition_heads

    model = load_model("gpt2-small")

    head_dla_pos, head_dla_neg = per_head_dla(model, pos_prompt, neg_prompt, target_token)
    # Shapes: (n_layers, n_heads) each

    top = top_inhibition_heads(head_dla_pos, head_dla_neg, top_k=10)
    # Returns list of ((layer, head), delta_dla) sorted by delta descending
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Core DLA function
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_head_dla(
    model: HookedTransformer,
    positive_prompt: str,
    negated_prompt: str,
    target_token: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-head DLA for both prompts.

    TransformerLens ``cache.stack_head_results()`` returns the outputs of
    every head projected back into residual-stream space.  We then dot each
    head's output with the unembedding vector for the target token.

    Parameters
    ----------
    model : HookedTransformer
    positive_prompt : str
    negated_prompt : str
    target_token : str
        Single token with leading space, e.g. ``" Paris"``.

    Returns
    -------
    head_dla_pos : np.ndarray  shape (n_layers, n_heads)
    head_dla_neg : np.ndarray  shape (n_layers, n_heads)
    """
    target_id   = model.to_single_token(target_token)
    W_U_target  = model.W_U[:, target_id]   # (d_model,)
    n_layers    = model.cfg.n_layers
    n_heads     = model.cfg.n_heads

    # Run with full cache to get per-head outputs
    _, pos_cache = model.run_with_cache(positive_prompt)
    _, neg_cache = model.run_with_cache(negated_prompt)

    # stack_head_results → (n_layers * n_heads, batch, seq_len, d_model)
    pos_head_out = pos_cache.stack_head_results(
        layer=-1, return_labels=False
    )  # shape: (n_layers * n_heads, batch, seq_len, d_model)
    neg_head_out = neg_cache.stack_head_results(
        layer=-1, return_labels=False
    )

    # Slice last token position → (n_layers * n_heads, d_model)
    pos_last = pos_head_out[:, 0, -1, :]  # (n_layers*n_heads, d_model)
    neg_last = neg_head_out[:, 0, -1, :]

    # DLA = head_output · W_U_target
    pos_dla_flat = (pos_last @ W_U_target).cpu().numpy()  # (n_layers*n_heads,)
    neg_dla_flat = (neg_last @ W_U_target).cpu().numpy()

    head_dla_pos = pos_dla_flat.reshape(n_layers, n_heads)
    head_dla_neg = neg_dla_flat.reshape(n_layers, n_heads)

    return head_dla_pos, head_dla_neg


# ---------------------------------------------------------------------------
# Identify top inhibition heads
# ---------------------------------------------------------------------------

def top_inhibition_heads(
    head_dla_pos: np.ndarray,
    head_dla_neg: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Identify the heads with the largest decrease in DLA from positive → negated.

    A large (DLA_pos - DLA_neg) means the head promotes the target on the
    positive prompt but suppresses it on the negated prompt → good inhibition.

    Parameters
    ----------
    head_dla_pos : ndarray (n_layers, n_heads)
    head_dla_neg : ndarray (n_layers, n_heads)
    top_k : int

    Returns
    -------
    List of ((layer, head), delta_dla) sorted by delta descending.
    """
    delta = head_dla_pos - head_dla_neg  # positive = more inhibition
    n_layers, n_heads = delta.shape
    flat_indices = np.argsort(delta.flatten())[::-1][:top_k]
    results = []
    for flat_idx in flat_indices:
        layer = int(flat_idx // n_heads)
        head  = int(flat_idx %  n_heads)
        results.append(((layer, head), float(delta[layer, head])))
    return results


def select_top_heads(delta: np.ndarray, top_k: int = 10) -> List[Tuple[int, int]]:
    """Return the top-k ``(layer, head)`` pairs from a delta matrix."""
    _, n_heads = delta.shape
    flat_indices = np.argsort(delta.reshape(-1))[::-1][:top_k]
    return [(int(flat_idx // n_heads), int(flat_idx % n_heads)) for flat_idx in flat_indices]


# ---------------------------------------------------------------------------
# Batch version over a DataFrame of results
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_head_dla_batch(
    model: HookedTransformer,
    pairs,                   # list of PromptPair
    top_k: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run :func:`per_head_dla` on each pair and accumulate a mean *delta* matrix.

    Returns a (n_layers, n_heads) array of mean (DLA_pos - DLA_neg),
    identifying the heads that most consistently inhibit the target after
    negation across the dataset.

    Parameters
    ----------
    model : HookedTransformer
    pairs : list of PromptPair
    top_k : int  — printed in the summary
    verbose : bool

    Returns
    -------
    mean_delta : np.ndarray  (n_layers, n_heads)
    """
    from tqdm import tqdm
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    accumulator = np.zeros((n_layers, n_heads))
    count = 0

    iterator = tqdm(pairs, desc="Per-head DLA") if verbose else pairs

    for pair in iterator:
        try:
            pos_dla, neg_dla = per_head_dla(
                model, pair.positive_prompt, pair.negated_prompt, pair.target_token
            )
            accumulator += (pos_dla - neg_dla)
            count += 1
        except Exception:
            continue

    if count == 0:
        return accumulator

    mean_delta = accumulator / count

    if verbose:
        print(f"\nTop-{top_k} inhibition heads (mean Δ DLA, positive→negated):")
        print(f"  {'Head (L,H)':<15} {'Mean ΔDLA':>10}")
        print("  " + "-" * 28)
        for layer, head in select_top_heads(mean_delta, top_k=top_k):
            print(f"  ({layer:2d}, {head:2d}){'':<8} {mean_delta[layer, head]:>10.4f}")

    return mean_delta


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_head_dla_heatmap(
    head_dla_pos: np.ndarray,
    head_dla_neg: np.ndarray,
    target_token: str = "",
    fig_dir: str = "figures",
    filename: str = "head_dla_heatmap.png",
) -> None:
    """
    Plot a 3-panel heatmap:
      left   — DLA on positive prompt
      centre — DLA on negated prompt
      right  — delta (pos - neg)
    """
    import seaborn as sns
    os.makedirs(fig_dir, exist_ok=True)

    delta = head_dla_pos - head_dla_neg
    vmax  = max(np.abs(head_dla_pos).max(), np.abs(head_dla_neg).max(), 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    kw = dict(center=0, cmap="RdBu_r", cbar=True, linewidths=0.3)

    sns.heatmap(head_dla_pos, ax=axes[0], vmin=-vmax, vmax=vmax, **kw)
    axes[0].set_title(f"Head DLA — Positive\nTarget: '{target_token}'")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")

    sns.heatmap(head_dla_neg, ax=axes[1], vmin=-vmax, vmax=vmax, **kw)
    axes[1].set_title(f"Head DLA — Negated")
    axes[1].set_xlabel("Head")

    vmax_d = max(np.abs(delta).max(), 0.1)
    sns.heatmap(delta, ax=axes[2], center=0, vmin=-vmax_d, vmax=vmax_d,
                cmap="PRGn", cbar=True, linewidths=0.3)
    axes[2].set_title("ΔDLA (pos − neg)\n= inhibition head map")
    axes[2].set_xlabel("Head")

    plt.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
