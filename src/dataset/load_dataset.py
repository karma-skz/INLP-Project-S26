"""
src/dataset/load_dataset.py
============================
Loads the NeelNanda/counterfact-tracing dataset from HuggingFace and
constructs prompt pairs (positive + negated) suitable for the analysis
pipeline.

CounterFact entry relevant fields
-----------------------------------
  prompt        : fully-formatted factual prompt, e.g. "The capital of France is"
  subject       : the subject entity, e.g. "France"
  target_true   : the true next-token string, e.g. " Paris"
  target_false  : a plausible but wrong alternative
  case_id       : integer identifier

Negation strategy
------------------
We build the negated prompt by appending " not" to the positive prompt:
    "The capital of France is"  →  "The capital of France is not"

This mirrors the single-prompt exploration in transformerLenstest.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Optional

from datasets import load_dataset as hf_load_dataset


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PromptPair:
    """A single (positive, negated) prompt pair with associated metadata."""
    case_id: int
    subject: str
    positive_prompt: str
    negated_prompt: str
    target_token: str          # e.g. " Paris"  (leading space included)
    target_false: str          # plausible distractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_leading_space(token: str) -> str:
    """GPT-2 / Pythia tokenizers expect a leading space on bare words."""
    token = token.strip()
    if not token.startswith(" "):
        token = " " + token
    return token


def _build_negated_prompt(positive_prompt: str) -> str:
    """
    Append ' not' to the positive prompt.

    The positive prompts in CounterFact already end with a verb / copula
    (e.g., "is", "was born in"), so appending " not" produces a grammatical
    negation in most cases:

        "The capital of France is"         →  "The capital of France is not"
        "Marie Curie was born in"          →  "Marie Curie was born in not"
        "The Eiffel Tower is located in"   →  "The Eiffel Tower is located in not"

    For 'was born in' style prompts the result is slightly awkward but the
    mechanistic analysis is still valid (we are probing what the model predicts
    after the negation token, not evaluating grammaticality).
    """
    # Strip trailing whitespace so we don't double-space
    return positive_prompt.rstrip() + " not"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_counterfact(
    split: str = "train",
    max_samples: Optional[int] = None,
    skip_multi_token_targets: bool = True,
    model=None,                       # optional HookedTransformer, used to validate tokens
    verbose: bool = True,
) -> List[PromptPair]:
    """
    Load NeelNanda/counterfact-tracing from HuggingFace and return a list
    of :class:`PromptPair` objects.

    Parameters
    ----------
    split : str
        HuggingFace dataset split (usually "train" — the dataset has only one).
    max_samples : int, optional
        Cap the number of entries returned (useful for quick tests).
    skip_multi_token_targets : bool
        When *model* is provided, skip entries whose target_true tokenises
        to more than one token — DLA is defined only for single next-token
        targets.
    model : HookedTransformer, optional
        If provided, uses ``model.to_single_token`` to validate targets.
    verbose : bool
        Print progress information.

    Returns
    -------
    list of PromptPair
    """
    if verbose:
        print("Loading NeelNanda/counterfact-tracing …", end=" ", flush=True)

    raw = hf_load_dataset("NeelNanda/counterfact-tracing", split=split)

    if verbose:
        print(f"done ({len(raw)} entries).")

    pairs: List[PromptPair] = []
    skipped_multi = 0
    skipped_missing = 0

    for entry in raw:
        # ── Extract fields (handle both flat and nested schemas) ──────────
        try:
            case_id = int(entry.get("case_id", len(pairs)))
            subject = entry.get("subject", "")

            # The prompt field is already formatted in NeelNanda's version
            positive_prompt = entry.get("prompt", "").strip()

            # target_true may come as a string or a dict {"str": "...", "id": ...}
            raw_true = entry.get("target_true", "")
            if isinstance(raw_true, dict):
                raw_true = raw_true.get("str", "")
            target_token = _ensure_leading_space(str(raw_true))

            raw_false = entry.get("target_false", "")
            if isinstance(raw_false, dict):
                raw_false = raw_false.get("str", "")
            target_false = _ensure_leading_space(str(raw_false))

        except Exception:
            skipped_missing += 1
            continue

        if not positive_prompt or not target_token.strip():
            skipped_missing += 1
            continue

        # ── Optionally validate single-token target ───────────────────────
        if model is not None and skip_multi_token_targets:
            try:
                model.to_single_token(target_token)
            except Exception:
                skipped_multi += 1
                continue

        negated_prompt = _build_negated_prompt(positive_prompt)

        pairs.append(
            PromptPair(
                case_id=case_id,
                subject=subject,
                positive_prompt=positive_prompt,
                negated_prompt=negated_prompt,
                target_token=target_token,
                target_false=target_false,
            )
        )

        if max_samples is not None and len(pairs) >= max_samples:
            break

    if verbose:
        print(f"  → {len(pairs)} valid prompt pairs "
              f"(skipped {skipped_multi} multi-token, {skipped_missing} missing fields)")

    return pairs


# ---------------------------------------------------------------------------
# Streaming variant (memory-efficient for very large runs)
# ---------------------------------------------------------------------------

def stream_counterfact(
    split: str = "train",
    max_samples: Optional[int] = None,
    model=None,
    skip_multi_token_targets: bool = True,
) -> Iterator[PromptPair]:
    """
    Streaming version of :func:`load_counterfact` — yields one
    :class:`PromptPair` at a time without loading the full dataset into RAM.
    """
    raw = hf_load_dataset("NeelNanda/counterfact-tracing", split=split, streaming=True)
    count = 0

    for entry in raw:
        if max_samples is not None and count >= max_samples:
            return

        try:
            case_id = int(entry.get("case_id", count))
            subject = entry.get("subject", "")
            positive_prompt = entry.get("prompt", "").strip()

            raw_true = entry.get("target_true", "")
            if isinstance(raw_true, dict):
                raw_true = raw_true.get("str", "")
            target_token = _ensure_leading_space(str(raw_true))

            raw_false = entry.get("target_false", "")
            if isinstance(raw_false, dict):
                raw_false = raw_false.get("str", "")
            target_false = _ensure_leading_space(str(raw_false))
        except Exception:
            continue

        if not positive_prompt or not target_token.strip():
            continue

        if model is not None and skip_multi_token_targets:
            try:
                model.to_single_token(target_token)
            except Exception:
                continue

        yield PromptPair(
            case_id=case_id,
            subject=subject,
            positive_prompt=positive_prompt,
            negated_prompt=_build_negated_prompt(positive_prompt),
            target_token=target_token,
            target_false=target_false,
        )
        count += 1
