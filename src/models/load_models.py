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
from transformer_lens import loading_from_pretrained as tl_loading

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    # short alias  →  TransformerLens model name
    "gpt2-small":  "gpt2-small",
    "gpt2":        "gpt2-small",          # convenience alias
    "gpt2-medium": "gpt2-medium",
    "gpt2-large":  "gpt2-large",
    "pythia-160m": "pythia-160m",
    "pythia":      "pythia-160m",         # convenience alias
    "pythia-410m": "EleutherAI/pythia-410m",
}

# Canonical short names (used as keys in results dictionaries)
MODEL_SHORTNAMES: dict[str, str] = {
    "gpt2-small":  "gpt2-small",
    "gpt2":        "gpt2-small",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large":  "gpt2-large",
    "pythia-160m": "pythia-160m",
    "pythia":      "pythia-160m",
    "pythia-410m": "pythia-410m",
}

CANONICAL_MODEL_NAMES: list[str] = [
    "gpt2-small",
    "gpt2-medium",
    "gpt2-large",
    "pythia-160m",
    "pythia-410m",
]


def _patch_transformerlens_hf_compat() -> None:
    """
    Bridge minor config-schema differences between current transformers and
    the installed TransformerLens version.

    Newer GPT-NeoX configs store the partial rotary factor under
    ``rope_parameters.partial_rotary_factor`` instead of exposing
    ``rotary_pct`` directly. TransformerLens still expects ``rotary_pct``
    for Pythia-family models.
    """
    current = tl_loading.AutoConfig.from_pretrained
    if getattr(current, "_inlp_rotary_pct_patch", False):
        return

    def patched_from_pretrained(*args, **kwargs):
        cfg = current(*args, **kwargs)
        if cfg.__class__.__name__ == "GPTNeoXConfig" and not hasattr(cfg, "rotary_pct"):
            rope_params = getattr(cfg, "rope_parameters", None) or {}
            rotary_pct = rope_params.get("partial_rotary_factor", 0.25)
            setattr(cfg, "rotary_pct", rotary_pct)
        return cfg

    patched_from_pretrained._inlp_rotary_pct_patch = True  # type: ignore[attr-defined]
    tl_loading.AutoConfig.from_pretrained = patched_from_pretrained


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    model_name: str = "gpt2-small",
    device: str | None = None,
    dtype: str | None = None,
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

    _patch_transformerlens_hf_compat()

    tl_name = SUPPORTED_MODELS[model_name]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = "float16" if device == "cuda" else "float32"

    model = HookedTransformer.from_pretrained(
        tl_name,
        device=device,
        dtype=dtype,
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
            f"d_model={cfg.d_model}  device={device}  dtype={dtype}"
        )

    return model


def get_device(model: HookedTransformer) -> str:
    """Return the device string the model lives on."""
    return str(next(model.parameters()).device)
