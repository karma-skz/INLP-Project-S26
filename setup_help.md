# Cluster Run Instructions

If you have just cloned this repository and need to run the experiments on a GPU cluster (or Kaggle/Colab), follow these exact steps sequentially.

## 1. Environment Setup
Create a Conda environment and install the required dependencies.

```bash
# Create and activate a fresh environment
conda create -n neg-project python=3.10 -y
conda activate neg-project

# Install PyTorch (Modify the CUDA version to match your cluster, e.g., cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core mechanistic interpretability libraries
pip install transformer_lens sae_lens wandb scikit-learn datasets transformers
```

## 2. Authentication (HuggingFace & WandB)
You must authenticate to download the target models (like Meta-Llama-3-8B) and to track experiment metrics.

Set these directly in your terminal before running any Python scripts:

```bash
# 1. HuggingFace (Required for Llama-3/Gemma-2)
# Get this from: https://huggingface.co/settings/tokens
export HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN_HERE"

# 2. Weights & Biases (Required for logging)
# Get this from: https://wandb.ai/authorize
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
```

## 3. Verify Setup (Smoke Test)
Before launching a massive job or downloading datasets, run the smoke test to verify your GPU and authentication tokens are working perfectly:

```bash
python -m pytest tests/test_smoke.py -v
```
*(If this fails, do not proceed. Check the error message to fix your environment.)*

## 4. Download Datasets
The datasets (CounterFact, GSM8K, Negated LAMA, etc.) are **ignored by git** and must be fetched automatically to your `data/` directory.

```bash
# This script will download and extract all necessary datasets
python main.py generate-data --task download
```

## 5. Run the Experiments
With the environment authenticated and datasets downloaded, you can now run the core evaluation loops. The unified CLI handles model loading, synthetic triplet generation, and causal metric extraction.

### Standard Models (1 GPU Required)
These models require ~16-24GB VRAM (e.g., RTX 3090/4090, A10G, L4, or A100).

```bash
# 1. Factual Parity Evaluation
python main.py run-experiment --task factual --model meta-llama/Meta-Llama-3-8B
python main.py run-experiment --task factual --model google/gemma-2-9b

# 2. Reasoning Logic Evaluation
python main.py run-experiment --task reasoning --model meta-llama/Meta-Llama-3-8B
python main.py run-experiment --task reasoning --model google/gemma-2-9b

# 3. Causal Interventions (Activation Patching & CIS)
python main.py run-experiment --task intervention --model meta-llama/Meta-Llama-3-8B
python main.py run-experiment --task intervention --model google/gemma-2-9b
```

### Massive Models (Multi-GPU Required)
These models require >80GB VRAM (e.g., 2x to 4x A100 80GB). The script will automatically detect their size and fragment them across all available GPUs using HuggingFace Accelerate (`device_map="auto"`).

```bash
# 1. Factual Parity Evaluation
python main.py run-experiment --task factual --model meta-llama/Meta-Llama-3-70B
python main.py run-experiment --task factual --model google/gemma-2-27b

# 2. Reasoning Logic Evaluation
python main.py run-experiment --task reasoning --model meta-llama/Meta-Llama-3-70B
python main.py run-experiment --task reasoning --model google/gemma-2-27b

# 3. Causal Interventions
python main.py run-experiment --task intervention --model meta-llama/Meta-Llama-3-70B
python main.py run-experiment --task intervention --model google/gemma-2-27b
```
