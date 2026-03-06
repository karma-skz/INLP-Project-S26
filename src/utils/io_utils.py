"""
I/O utility functions for the INLP project.

Provides helpers for loading configs, saving results,
and other file operations.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        A dictionary of configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save benchmark results to a JSON file.

    Args:
        results: The results dictionary to save.
        output_path: Path to the output JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load benchmark results from a JSON file.

    Args:
        results_path: Path to the JSON results file.

    Returns:
        A dictionary of benchmark results.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    return results
