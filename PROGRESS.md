# Project Progress: Negation & Logical Parity in LLMs

This document tracks the current progress of the Mechanistic Interpretability project on Negation, outlining the architecture, datasets, and every experiment module built so far.

## 1. Datasets & Generation Pipelines (`src/dataset/`)
We have completely refactored the dataset generation into a unified `dataset_builder.py` factory, removing all fragmented legacy scripts and stubs.
- `download_datasets()`: Automates the downloading and caching of base datasets (CounterFact, GSM8K, Negated LAMA, CondaQA, Thunder-NUBench) cleanly into the local `data/` directory structure.
- `ParityGenerator`: Generates the core `Negation-Parity-Bench` using strict factual templates.
- **[CONSTRAINT NOTE] `ReasoningNegator`**: A logic-inversion module that transforms standard reasoning queries (like GSM8K) into logically negated variants. *Academic Constraint:* Currently implemented using a robust rule-based syntactic regex matcher to avoid the unreliability of stubs. For final publication runs on a live cluster, this should ideally be swapped for a rigorous dependency parser (e.g., `spaCy`) or an LLM call to ensure perfect semantic inversion of complex math constraints.
- `load_dataset.py`: Handles the ingestion and preprocessing of foundational factual datasets (e.g., CounterFact) used for baseline single-negation tests.

*All generated data is structured within the `data/factual/` and `data/reasoning/` directories.*

## 2. Core Mechanistic Analysis (`src/analysis/`)
We have completely implemented the core metrics, stripping out all placeholders and stubs in favor of unified modules.
- `causal_tracing.py`: Centralized hub for Activation Patching, Amplification, and calculating the **Compositional Interference Score (CIS)** via Direct Logit Attribution differences.
- `feature_extraction.py`: 
  - **Latent Truth-Value Toggling (LTVT)**: Implements `scikit-learn` Logistic Regression probes to classify the $P$ vs $\neg P$ probability mass dynamically across layers.
  - **SAE Tracker**: Integrates with the `sae_lens` library to correctly encode dense hidden states into sparse, interpretable "negation" features.
- `per_head.py`: Calculates per-head Direct Logit Attribution (DLA) across the transformer to track which components boost factual tokens (FFNs) and which suppress them (Inhibition Heads).

## 3. Benchmarking & Core Metrics (`src/benchmark/`)
- `run_benchmark.py`: Runs the core forward passes for the single-negation logic-memory conflict experiments.
- `sgr_analysis.py`: Computes the **Signal-to-Gate Ratio (SGR)**, which quantifies the physical tension between FFN memory retrieval (signal) and Attention suppression (gate).

## 4. Implemented Experiments (`src/experiments/`)
- `evaluator.py`: We have merged the fragmented benchmarking scripts into a single robust evaluator. It actively loads models via `TransformerLens`, orchestrates the triplet data generation, runs the full forward-pass evaluation loop, and computes both CIS and LTVT metrics over target parameters.

## 5. Central Orchestration & Utilities
- `main.py`: The unified `argparse`-based Command Line Interface (CLI) in the root directory. It serves as the primary entry point to trigger data generation pipelines and experiment runners across different tasks (factual, reasoning) and model scales.
- `models/load_models.py`: Centralized loading of gated HuggingFace weights using `TransformerLens`.
- `metrics/metrics.py`: Standardizes rank, probability, and accuracy calculations across all benchmarks.
- `utils/results.py`: Serialization of complex multi-dimensional outputs (Parquet, CSV) into standardized directories.
- `utils/plotting.py`: Matplotlib wrapper for standardizing visualizations.

## 6. Reporting & Visualization (`src/reporting/`)
- `visualize_trajectories.py`: Generates layer-by-layer line plots of the LTVT metric, visually demonstrating where the model gets trapped in the "Excluded Middle".
- `crossover.py`: Calculates and plots the "Crossover Layer" where inhibition DLA overtakes FFN retrieval DLA.
- `sgr_verification.py`: Validates that no negation failures occur mathematically when SGR $\leq 1$.
- `activation_patching.py` & `extended_amplification.py`: Generate dataset-scale plots for causal intervention results.

*All experiment outputs and visualizations are routed to the clean `results/factual/` and `results/reasoning/` directories.*

---

## 7. Next Steps
The core infrastructure, datasets, and experiment scripts are complete and documented. Moving forward, the immediate next steps are:

1. **Environment Provisioning**: Follow the instructions in `setup_help.md` to configure the GPU cluster (Conda, CUDA), authenticate with HuggingFace for Llama-3 access, and log in to Weights & Biases for tracking.
2. **Dataset Generation**: Use the central CLI to generate the full dataset (`python main.py generate-data --task download`).
3. **Run Baseline Experiments**: Launch the main evaluation loop on Llama-3-8B to establish the baseline failure rates for single and double negation (`python main.py run-experiment --task factual --model meta-llama/Meta-Llama-3-8B`).
4. **Causal Tracing**: Once baseline logs are verified in W&B, execute the causal tracing suite via the unified CLI (`python main.py run-experiment --task intervention --model meta-llama/Meta-Llama-3-8B`) to extract the Compositional Interference Score (CIS).
5. **Scaling Up**: Replicate the verified workflow on larger frontier models (`Llama-3-70B`, `Gemma-2-27B`) to test for scale-emergent behaviors.
