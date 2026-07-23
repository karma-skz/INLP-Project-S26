import os
import sys
import pytest
from pathlib import Path

def test_imports_and_dependencies():
    """Verify that all heavy compute and mechanistic libraries are installed."""
    try:
        import torch
        assert torch.__version__ is not None, "PyTorch is installed"
    except ImportError:
        pytest.fail("PyTorch is not installed. Required for TransformerLens.")

    try:
        import transformer_lens
    except ImportError:
        pytest.fail("transformer_lens is not installed. Run: pip install transformer_lens")

    try:
        import sae_lens
    except ImportError:
        pytest.fail("sae_lens is not installed. Required for SAE Feature tracking.")
        
    try:
        import wandb
    except ImportError:
        pytest.fail("wandb is not installed.")

    try:
        import sklearn
    except ImportError:
        pytest.fail("scikit-learn is not installed. Required for LTVT Logistic Regression probes.")

def test_huggingface_auth():
    """
    Verify HuggingFace authentication. 
    Required to download gated models like Meta-Llama-3-8B and Gemma-2-9B.
    """
    # Check if token is in environment
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Check if logged in via CLI
    cli_token_path = Path.home() / ".cache" / "huggingface" / "token"
    
    has_auth = (token is not None) or cli_token_path.exists()
    
    if not has_auth:
        pytest.fail(
            "No HuggingFace authentication found. \n"
            "Action Required: Run 'huggingface-cli login' OR set HF_TOKEN environment variable."
        )

def test_wandb_auth():
    """Verify Weights & Biases authentication for logging."""
    token = os.environ.get("WANDB_API_KEY")
    cli_token_path = Path.home() / ".netrc"
    
    has_auth = (token is not None) or (cli_token_path.exists() and "api.wandb.ai" in cli_token_path.read_text())
    
    if not has_auth:
        pytest.fail(
            "No WandB authentication found. \n"
            "Action Required: Run 'wandb login' OR set WANDB_API_KEY environment variable."
        )

def test_dataset_generation_logic():
    """Verify that the parity dataset generator compiles and functions correctly."""
    from src.dataset.dataset_builder import ParityGenerator
    generator = ParityGenerator()
    dummy_facts = [{"subject": "France", "relation": "capital", "target": " Paris"}]
    triplets = generator.generate_factual_triplets(dummy_facts)
    
    assert len(triplets) == 1
    assert triplets[0].P == "The capital of France is"
    assert triplets[0].not_P == "The capital of France is not"
    assert triplets[0].not_not_P == "The capital of France is not not"

def test_gpu_availability():
    """Warns if GPU is not available, as experiments require significant compute."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Ensure you are running this on a GPU cluster node, not a login node.")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
