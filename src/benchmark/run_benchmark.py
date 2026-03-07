"""
src/benchmark/run_benchmark.py
================================
Runs the DLA + SGR analysis pipeline over the full CounterFact dataset
(or any list of PromptPairs) and collects results into a CSV-ready DataFrame.

Activation patching is available per-sample but disabled by default because
it multiplies runtime by ~n_layers.  Turn it on with ``run_patching=True``
for a smaller sample (e.g. first 200 entries).

Usage
-----
    from src.models import load_model
    from src.dataset import load_counterfact
    from src.benchmark import run_benchmark

    model  = load_model("gpt2-small")
    pairs  = load_counterfact(max_samples=500, model=model)
    df     = run_benchmark(model, pairs, output_csv="results/gpt2_benchmark.csv")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.dataset.load_dataset import PromptPair


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    # --- identity ---
    case_id:          int
    subject:          str
    model_name:       str
    positive_prompt:  str
    negated_prompt:   str
    target_token:     str

    # --- behavioural ---
    pos_target_logit: float
    pos_target_prob:  float
    pos_target_rank:  int
    neg_target_logit: float
    neg_target_prob:  float
    neg_target_rank:  int
    negation_failure: bool    # True when negated rank < positive rank (model ignores negation)

    # --- aggregate DLA ---
    ffn_pos_total:    float
    ffn_neg_total:    float
    attn_pos_total:   float
    attn_neg_total:   float

    # --- SGR ---
    retrieval_strength:   float   # sum of positive FFN DLA on negated prompt
    inhibition_strength:  float   # |sum of negative Attn DLA on negated prompt|
    sgr:                  float   # retrieval_strength / inhibition_strength

    # --- crossover ---
    crossover_layer: Optional[int]   # None if no crossover

    # --- per-layer arrays stored as pipe-separated strings for CSV compat ---
    ffn_dla_pos_str:  str    # n_layers floats, e.g. "0.12|0.33|…"
    ffn_dla_neg_str:  str
    attn_dla_pos_str: str
    attn_dla_neg_str: str

    def ffn_dla_pos(self) -> np.ndarray:
        return np.array([float(x) for x in self.ffn_dla_pos_str.split("|")])

    def ffn_dla_neg(self) -> np.ndarray:
        return np.array([float(x) for x in self.ffn_dla_neg_str.split("|")])

    def attn_dla_pos(self) -> np.ndarray:
        return np.array([float(x) for x in self.attn_dla_pos_str.split("|")])

    def attn_dla_neg(self) -> np.ndarray:
        return np.array([float(x) for x in self.attn_dla_neg_str.split("|")])


def _arr_to_str(arr: np.ndarray) -> str:
    return "|".join(f"{v:.5f}" for v in arr)


# ---------------------------------------------------------------------------
# Core per-sample analysis (no patching)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _analyse_pair(
    model: HookedTransformer,
    pair: PromptPair,
    model_name: str,
) -> Optional[BenchmarkResult]:
    """
    Run DLA + SGR analysis on a single PromptPair.
    Returns None if the target cannot be represented as a single token.
    """
    # ── target token ────────────────────────────────────────────────────────
    try:
        target_id = model.to_single_token(pair.target_token)
    except Exception:
        return None  # skip multi-token targets

    n_layers = model.cfg.n_layers
    W_U_target = model.W_U[:, target_id]  # (d_model,)

    # ── run both prompts with full cache ─────────────────────────────────────
    pos_logits, pos_cache = model.run_with_cache(pair.positive_prompt)
    neg_logits, neg_cache = model.run_with_cache(pair.negated_prompt)

    # ── behavioural stats ────────────────────────────────────────────────────
    def _stats(logits_batch):
        logits = logits_batch[0, -1, :]
        probs  = torch.softmax(logits, dim=-1)
        logit  = logits[target_id].item()
        prob   = probs[target_id].item()
        rank   = int((logits >= logits[target_id]).sum().item())
        return logit, prob, rank

    pos_logit, pos_prob, pos_rank = _stats(pos_logits)
    neg_logit, neg_prob, neg_rank = _stats(neg_logits)
    negation_failure = neg_rank < pos_rank

    # ── DLA ─────────────────────────────────────────────────────────────────
    pos_resid, pos_labels = pos_cache.decompose_resid(return_labels=True, mode="full")
    neg_resid, neg_labels = neg_cache.decompose_resid(return_labels=True, mode="full")

    pos_dla = (pos_resid[:, 0, -1, :] @ W_U_target).cpu().numpy()
    neg_dla = (neg_resid[:, 0, -1, :] @ W_U_target).cpu().numpy()

    # ── per-layer FFN / Attn split ───────────────────────────────────────────
    pl_ffn_pos  = np.zeros(n_layers)
    pl_ffn_neg  = np.zeros(n_layers)
    pl_attn_pos = np.zeros(n_layers)
    pl_attn_neg = np.zeros(n_layers)

    for i, label in enumerate(pos_labels):
        parts = label.split("_")
        if len(parts) >= 2 and parts[0].isdigit():
            layer = int(parts[0])
            if "mlp" in label.lower():
                pl_ffn_pos[layer]  += pos_dla[i]
                pl_ffn_neg[layer]  += neg_dla[i]
            elif "attn" in label.lower():
                pl_attn_pos[layer] += pos_dla[i]
                pl_attn_neg[layer] += neg_dla[i]

    # ── aggregate totals ─────────────────────────────────────────────────────
    ffn_pos_total  = float(pl_ffn_pos.sum())
    ffn_neg_total  = float(pl_ffn_neg.sum())
    attn_pos_total = float(pl_attn_pos.sum())
    attn_neg_total = float(pl_attn_neg.sum())

    # ── SGR ──────────────────────────────────────────────────────────────────
    retrieval    = float(pl_ffn_neg[pl_ffn_neg > 0].sum())
    inhibition   = float(abs(pl_attn_neg[pl_attn_neg < 0].sum()))
    sgr          = retrieval / inhibition if inhibition > 0 else float("inf")

    # ── crossover ────────────────────────────────────────────────────────────
    cum_ffn   = np.cumsum(pl_ffn_neg)
    cum_attn  = np.cumsum(pl_attn_neg)
    cum_total = cum_ffn + cum_attn
    crossover: Optional[int] = None
    for layer in range(1, n_layers):
        if cum_total[layer - 1] < 0 <= cum_total[layer]:
            crossover = layer
            break

    return BenchmarkResult(
        case_id=pair.case_id,
        subject=pair.subject,
        model_name=model_name,
        positive_prompt=pair.positive_prompt,
        negated_prompt=pair.negated_prompt,
        target_token=pair.target_token,
        pos_target_logit=pos_logit,
        pos_target_prob=pos_prob,
        pos_target_rank=pos_rank,
        neg_target_logit=neg_logit,
        neg_target_prob=neg_prob,
        neg_target_rank=neg_rank,
        negation_failure=negation_failure,
        ffn_pos_total=ffn_pos_total,
        ffn_neg_total=ffn_neg_total,
        attn_pos_total=attn_pos_total,
        attn_neg_total=attn_neg_total,
        retrieval_strength=retrieval,
        inhibition_strength=inhibition,
        sgr=sgr,
        crossover_layer=crossover,
        ffn_dla_pos_str=_arr_to_str(pl_ffn_pos),
        ffn_dla_neg_str=_arr_to_str(pl_ffn_neg),
        attn_dla_pos_str=_arr_to_str(pl_attn_pos),
        attn_dla_neg_str=_arr_to_str(pl_attn_neg),
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    model: HookedTransformer,
    pairs: List[PromptPair],
    model_name: str = "gpt2-small",
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the DLA + SGR analysis over *pairs* and return a DataFrame.

    One row per PromptPair.  Per-layer DLA arrays are stored as
    pipe-separated strings in columns ``ffn_dla_pos_str``, etc.

    Parameters
    ----------
    model : HookedTransformer
        Loaded model (use ``src.models.load_model``).
    pairs : list of PromptPair
        Prompt pairs to analyse (from ``src.dataset.load_counterfact``).
    model_name : str
        Short identifier stored in the ``model_name`` column.
    output_csv : str, optional
        If provided, save the DataFrame to this path (creates dirs).
    verbose : bool
        Show tqdm progress bar and summary stats.

    Returns
    -------
    pd.DataFrame
    """
    results: List[BenchmarkResult] = []
    skipped = 0

    iterator = tqdm(pairs, desc=f"Benchmarking {model_name}") if verbose else pairs

    for pair in iterator:
        result = _analyse_pair(model, pair, model_name)
        if result is None:
            skipped += 1
            continue
        results.append(result)

    if verbose:
        n = len(results)
        failures = sum(r.negation_failure for r in results)
        avg_sgr  = np.mean([r.sgr for r in results if r.sgr != float("inf")])
        print(
            f"\n{'='*60}\n"
            f"Benchmark complete — {n} samples  ({skipped} skipped)\n"
            f"  Negation failure rate : {failures/n:.1%}  ({failures}/{n})\n"
            f"  Mean SGR (finite)     : {avg_sgr:.3f}\n"
            f"{'='*60}"
        )

    df = pd.DataFrame([asdict(r) for r in results])

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"Results saved → {output_csv}")

    return df
