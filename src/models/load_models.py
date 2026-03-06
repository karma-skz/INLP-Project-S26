"""
Model loader using TransformerLens.

Provides lazy loading of HookedTransformer models with automatic
GPU placement for mechanistic interpretability experiments.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict, Optional


# Global cache for loaded models (lazy loading)
_model_cache: Dict[str, HookedTransformer] = {}


def get_device(device: str = "auto") -> str:
    """Determine the device to use for model inference.

    Args:
        device: Device specification. Use 'auto' for automatic detection,
            or specify 'cuda', 'cpu', etc.

    Returns:
        The resolved device string.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(
    model_name: str,
    device: str = "auto",
    use_cache: bool = True,
) -> HookedTransformer:
    """Load a HookedTransformer model with lazy caching.

    Loads the specified model using TransformerLens and moves it to
    the appropriate device. Models are cached after first load to
    avoid redundant downloads.

    Args:
        model_name: Name of the model to load. Supported models:
            - 'gpt2-small'
            - 'pythia-160m'
        device: Device to load the model onto. Use 'auto' for
            automatic GPU/CPU detection.
        use_cache: Whether to cache and reuse loaded models.

    Returns:
        A HookedTransformer instance ready for inference.

    Example:
        >>> model = load_model("gpt2-small")
        >>> model = load_model("pythia-160m", device="cpu")
    """
    if use_cache and model_name in _model_cache:
        return _model_cache[model_name]

    resolved_device = get_device(device)
    print(f"Loading model '{model_name}' on device '{resolved_device}'...")

    model = HookedTransformer.from_pretrained(
        model_name,
        device=resolved_device,
    )

    if use_cache:
        _model_cache[model_name] = model

    print(f"Model '{model_name}' loaded successfully.")
    return model


def clear_model_cache() -> None:
    """Clear all cached models from memory."""
    global _model_cache
    _model_cache.clear()
    torch.cuda.empty_cache()
