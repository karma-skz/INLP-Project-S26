"""
Metrics for evaluating negation handling in LLMs.

Provides functions to compute accuracy, negation failure rate,
and extract target token logits from model outputs.
"""

import torch
from transformer_lens import HookedTransformer
from typing import List, Dict


def extract_target_logit(
    model: HookedTransformer,
    prompt: str,
    target: str,
) -> float:
    """Extract the logit value for the target token given a prompt.

    Args:
        model: A HookedTransformer model instance.
        prompt: The input prompt string.
        target: The target completion token.

    Returns:
        The logit value (float) assigned to the target token
        at the last position of the prompt.
    """
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    # Get logits at the last token position
    last_logits = logits[0, -1, :]

    # Encode the target token
    target_token_id = model.to_single_token(target)
    target_logit = last_logits[target_token_id].item()

    return target_logit


def get_top_prediction(
    model: HookedTransformer,
    prompt: str,
) -> str:
    """Get the top predicted token for a given prompt.

    Args:
        model: A HookedTransformer model instance.
        prompt: The input prompt string.

    Returns:
        The decoded string of the top-1 predicted token.
    """
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    last_logits = logits[0, -1, :]

    top_token_id = torch.argmax(last_logits).item()
    top_token = model.to_string(top_token_id)

    return top_token


def compute_accuracy(results: List[Dict]) -> float:
    """Compute positive accuracy from benchmark results.

    Positive accuracy is the fraction of samples where the model's
    top prediction on the positive prompt matches the target token.

    Args:
        results: A list of benchmark result dictionaries, each containing
            'positive_prediction' and 'target' keys.

    Returns:
        The accuracy as a float between 0.0 and 1.0.
    """
    if not results:
        return 0.0

    correct = sum(
        1 for r in results
        if r["positive_prediction"].strip().lower() == r["target"].strip().lower()
    )
    return correct / len(results)


def compute_negation_failure_rate(results: List[Dict]) -> float:
    """Compute the negation failure rate from benchmark results.

    Negation failure occurs when the model still predicts the factual
    target token even after the prompt has been negated.

    Args:
        results: A list of benchmark result dictionaries, each containing
            'negated_prediction' and 'target' keys.

    Returns:
        The negation failure rate as a float between 0.0 and 1.0.
    """
    if not results:
        return 0.0

    failures = sum(
        1 for r in results
        if r["negated_prediction"].strip().lower() == r["target"].strip().lower()
    )
    return failures / len(results)
