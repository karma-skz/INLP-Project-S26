"""
Prompt builder for negation experiments.

Converts factual prompts into positive and negated versions
for benchmarking negation failure rates in LLMs.
"""

from typing import Dict, List, Tuple


def build_positive_prompt(prompt: str) -> str:
    """Return the prompt as-is (positive / factual form).

    Args:
        prompt: The original factual prompt.

    Returns:
        The unchanged prompt string.
    """
    return prompt.strip()


def build_negated_prompt(prompt: str) -> str:
    """Append ' not' to the prompt to create a negated version.

    Args:
        prompt: The original factual prompt.

    Returns:
        The negated prompt string with ' not' appended.

    Example:
        >>> build_negated_prompt("The capital of France is")
        'The capital of France is not'
    """
    return prompt.strip() + " not"


def build_prompt_pair(entry: Dict[str, str]) -> Dict[str, str]:
    """Build a positive/negated prompt pair from a dataset entry.

    Args:
        entry: A dictionary with 'prompt' and 'target' keys.

    Returns:
        A dictionary containing:
            - "positive_prompt": The original factual prompt.
            - "negated_prompt": The negated version of the prompt.
            - "target": The expected target token.
    """
    return {
        "positive_prompt": build_positive_prompt(entry["prompt"]),
        "negated_prompt": build_negated_prompt(entry["prompt"]),
        "target": entry["target"],
    }


def build_all_prompt_pairs(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build positive/negated prompt pairs for an entire dataset.

    Args:
        data: A list of dataset entries, each with 'prompt' and 'target' keys.

    Returns:
        A list of dictionaries, each containing positive_prompt,
        negated_prompt, and target fields.
    """
    return [build_prompt_pair(entry) for entry in data]
