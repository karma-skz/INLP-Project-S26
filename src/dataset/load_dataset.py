"""
Dataset loader for the CounterFact dataset.

Downloads and processes the NeelNanda/counterfact-tracing dataset
from HuggingFace for negation failure analysis.
"""

from datasets import load_dataset as hf_load_dataset
from typing import List, Dict, Optional


def load_counterfact(dataset_size: Optional[int] = None) -> List[Dict[str, str]]:
    """Load the CounterFact dataset from HuggingFace.

    Downloads the NeelNanda/counterfact-tracing dataset and returns
    a list of entries with prompt and target fields.

    Args:
        dataset_size: Maximum number of samples to load.
            If None, loads the entire dataset.

    Returns:
        A list of dictionaries, each containing:
            - "prompt": The factual prompt string.
            - "target": The expected target completion.

    Example:
        >>> data = load_counterfact(dataset_size=10)
        >>> data[0]
        {'prompt': 'The capital of France is', 'target': 'Paris'}
    """
    dataset = hf_load_dataset("NeelNanda/counterfact-tracing", split="train")

    entries = []
    for i, item in enumerate(dataset):
        if dataset_size is not None and i >= dataset_size:
            break

        entry = {
            "prompt": item["prompt"],
            "target": item["target"],
        }
        entries.append(entry)

    return entries
