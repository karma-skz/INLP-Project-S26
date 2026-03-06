"""
Benchmark runner for negation failure analysis.

Iterates through dataset samples, runs both positive and negated prompts
through the model, and collects predictions and logit information.
"""

import torch
from tqdm import tqdm
from typing import List, Dict

from transformer_lens import HookedTransformer

from src.benchmark.metrics import (
    extract_target_logit,
    get_top_prediction,
    compute_accuracy,
    compute_negation_failure_rate,
)
from src.dataset.build_prompts import build_all_prompt_pairs


def run_benchmark(
    model: HookedTransformer,
    data: List[Dict[str, str]],
    model_name: str = "",
) -> Dict:
    """Run the negation failure benchmark on a model.

    For each sample in the dataset, runs both the positive and negated
    prompts through the model and records predictions and target logits.

    Args:
        model: A HookedTransformer model instance.
        data: A list of dataset entries with 'prompt' and 'target' keys.
        model_name: Name of the model (for display purposes).

    Returns:
        A dictionary containing:
            - "model": The model name.
            - "positive_accuracy": Fraction of correct positive predictions.
            - "negation_failure_rate": Fraction of negation failures.
            - "results": List of per-sample result dictionaries.
    """
    prompt_pairs = build_all_prompt_pairs(data)
    results = []

    desc = f"Benchmarking {model_name}" if model_name else "Benchmarking"

    for pair in tqdm(prompt_pairs, desc=desc):
        positive_prompt = pair["positive_prompt"]
        negated_prompt = pair["negated_prompt"]
        target = pair["target"]

        with torch.no_grad():
            # Get predictions
            positive_pred = get_top_prediction(model, positive_prompt)
            negated_pred = get_top_prediction(model, negated_prompt)

            # Get target logits
            target_logit_pos = extract_target_logit(
                model, positive_prompt, target
            )
            target_logit_neg = extract_target_logit(
                model, negated_prompt, target
            )

        result = {
            "positive_prompt": positive_prompt,
            "negated_prompt": negated_prompt,
            "target": target,
            "positive_prediction": positive_pred,
            "negated_prediction": negated_pred,
            "target_logit_positive": target_logit_pos,
            "target_logit_negated": target_logit_neg,
        }
        results.append(result)

    # Compute aggregate metrics
    positive_accuracy = compute_accuracy(results)
    negation_failure_rate = compute_negation_failure_rate(results)

    print(f"\n--- {model_name} Results ---")
    print(f"  Positive Accuracy:      {positive_accuracy:.4f}")
    print(f"  Negation Failure Rate:  {negation_failure_rate:.4f}")

    return {
        "model": model_name,
        "positive_accuracy": positive_accuracy,
        "negation_failure_rate": negation_failure_rate,
        "results": results,
    }
