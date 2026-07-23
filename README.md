# Deciphering the Compositional Collapse: Negation in LLMs

This repository houses a mechanistic interpretability study on **Negation and Logical Parity** in Large Language Models. 

While LLMs often struggle with single negation ($\neg P$), they experience a profound "compositional collapse" when processing double negation ($\neg\neg P$). We utilize `TransformerLens`, Sparse Autoencoders (SAEs), and novel causal metrics to pinpoint exactly where this logic-memory conflict breaks down in the residual stream.

## Key Contributions & Metrics
- **The Competitive Gating Hypothesis:** We frame negation as a physical conflict between early-layer FFN memory retrieval (the signal) and late-layer Attention suppression (the gate), measured by the **Signal-to-Gate Ratio (SGR)**.
- **Latent Truth-Value Toggling (LTVT):** Linear probes tracking the exact layer where the model state flips (or gets trapped in the "Excluded Middle").
- **Compositional Interference Score (CIS):** A causal metric tracking how a secondary negator actively attenuates the primary negator's inhibition circuit.

## Directory Structure
The repository has been fully modularized and optimized for scale:
- `data/` and `results/`: Systematically split into `factual/` and `reasoning/` tasks.
- `src/dataset/dataset_builder.py`: A unified factory for downloading base datasets (CounterFact, GSM8K, Negated LAMA) and generating parity triplets.
- `src/analysis/`:
  - `causal_tracing.py`: Centralized Activation Patching, Amplification, and CIS calculations.
  - `feature_extraction.py`: LTVT probe training (via `scikit-learn`) and SAE extraction (via `sae_lens`).
- `src/experiments/evaluator.py`: The robust evaluation loop driving the metrics on models like Llama-3-8B.
- `main.py`: The central CLI orchestrating all runs.

## Setup & Execution
Please refer to `setup_help.md` for cluster configuration and Weights & Biases (W&B) integration.

**1. Download & Generate Datasets:**
```bash
python main.py generate-data --task download
python main.py generate-data --task factual
```

**2. Run Evaluation Loop:**
```bash
python main.py run-experiment --task factual --model meta-llama/Meta-Llama-3-8B
```
