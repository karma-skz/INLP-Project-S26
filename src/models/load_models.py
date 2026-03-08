"""
src/models/load_models.py
==========================
Thin wrapper around TransformerLens's HookedTransformer for the two
models used in this project.

Supported models
----------------
  gpt2-small    — 117M / 124M parameter GPT-2 (12 layers, 12 heads)
  pythia-160m   — EleutherAI Pythia 160M      (12 layers, 12 heads)

Usage
-----
    from src.models import load_model

    model = load_model("gpt2-small")          # default
    model = load_model("pythia-160m")
    model = load_model("gpt2-small", device="cuda")
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    # short alias  →  TransformerLens model name
    "gpt2-small":  "gpt2-small",
    "gpt2":        "gpt2-small",          # convenience alias
    "pythia-160m": "pythia-160m",
    "pythia":      "pythia-160m",         # convenience alias
}

# Canonical short names (used as keys in results dictionaries)
MODEL_SHORTNAMES: dict[str, str] = {
    "gpt2-small":  "gpt2-small",
    "gpt2":        "gpt2-small",
    "pythia-160m": "pythia-160m",
    "pythia":      "pythia-160m",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    model_name: str = "gpt2-small",
    device: str | None = None,
    fold_ln: bool = True,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
    verbose: bool = True,
) -> HookedTransformer:
    """
    Load a HookedTransformer model by short name.

    Parameters
    ----------
    model_name : str
        One of ``"gpt2-small"``, ``"gpt2"``, ``"pythia-160m"``, ``"pythia"``.
    device : str, optional
        ``"cpu"``, ``"cuda"``, or ``"mps"``.  Auto-detected when ``None``.
    fold_ln : bool
        Fold LayerNorm parameters into surrounding weights (makes DLA exact).
    center_writing_weights : bool
        Subtract the mean from weight matrices — stabilises DLA numerics.
    center_unembed : bool
        Centre the unembedding matrix (standard for DLA analysis).
    verbose : bool
        Print a one-line summary after loading.

    Returns
    -------
    HookedTransformer
        Ready-to-use model on the chosen device.
    """
    if model_name not in SUPPORTED_MODELS:
        supported = list(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {supported}"
        )

    tl_name = SUPPORTED_MODELS[model_name]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(
        tl_name,
        device=device,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
    )
    # Enable per-head result hooks (must be set after loading in newer
    # TransformerLens/transformers versions to avoid leaking into HF loader)
    model.cfg.use_attn_result = True
    model.setup()
    model.eval()

    if verbose:
        cfg = model.cfg
        print(
            f"Loaded  {tl_name:<20}  "
            f"layers={cfg.n_layers}  heads={cfg.n_heads}  "
            f"d_model={cfg.d_model}  device={device}"
        )

    return model


def get_device(model: HookedTransformer) -> str:
    """Return the device string the model lives on."""
    return str(next(model.parameters()).device)
