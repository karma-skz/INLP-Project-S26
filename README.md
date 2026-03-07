# INLP Project: Negation Failures in Large Language Models

## Project Overview

This project studies **negation failures in large language models (LLMs)** — the "Pink Elephant" effect — using mechanistic interpretability tools (TransformerLens).

Large language models often fail to correctly handle negation. For example, when prompted with _"The capital of France is not"_, GPT-2 still assigns high probability to _"Paris"_. This project investigates **why** this happens at the circuit level by decomposing the residual stream, attributing logits to individual components, and applying causal interventions.

### Supported Models

- **GPT-2 Small** (`gpt2-small`) — 124M parameters
- **Pythia-160M** (`pythia-160m`) — 160M parameters

### Dataset

- [NeelNanda/counterfact-tracing](https://huggingface.co/datasets/NeelNanda/counterfact-tracing) via HuggingFace Datasets

---

## Repository Structure

```
INLP-Project-S26/
├── README.md
├── requirements.txt
├── environment.yml
├── explanation.md                  # Detailed pipeline walkthrough
├── transformerLenstest.py          # Single-prompt exploration (original prototype)
├── run_pipeline.py                 # Full dataset pipeline entry point
├── results/                        # CSV outputs from benchmark runs
│   ├── gpt2-small_benchmark.csv
│   └── all_models_benchmark.csv
├── figures/                        # All generated figures
│   ├── 01_ffn_dla_comparison.png
│   ├── 02_attn_dla_comparison.png
│   ├── 03_cumulative_dla_crossover.png
│   ├── 04_activation_patching.png
│   ├── 05_sgr.png
│   ├── sgr_histogram.png
│   ├── sgr_failure_rate.png
│   ├── per_layer_dla_mean.png
│   ├── head_dla_heatmap.png
│   ├── amplification_sweep.png
│   └── amplification_failure_rate.png
└── src/
    ├── dataset/
    │   └── load_dataset.py         # Loads CounterFact, builds PromptPair objects
    ├── models/
    │   └── load_models.py          # Model loader (gpt2-small, pythia-160m)
    ├── benchmark/
    │   ├── run_benchmark.py        # DLA + SGR over full dataset → CSV
    │   └── sgr_analysis.py         # SGR distribution figures
    ├── analysis/
    │   ├── per_head.py             # Per-attention-head DLA, inhibition head detection
    │   └── amplification.py        # Artificial amplification experiment
    └── metrics/
        └── metrics.py              # Statistical tests (Spearman, Mann-Whitney, CIs)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/karma-skz/INLP-Project-S26.git
cd INLP-Project-S26
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate inlp-project
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Full dataset pipeline (recommended)

```bash
# GPT-2 Small, 200 samples (default — fast)
python run_pipeline.py

# Both models, full dataset
python run_pipeline.py --models gpt2-small pythia-160m --max_samples -1

# Skip slow stages (useful for quick re-runs)
python run_pipeline.py --skip_per_head --skip_amplification
```

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--models` | `gpt2-small` | Space-separated list of models to benchmark |
| `--max_samples` | `200` | CounterFact samples per model (`-1` = all 21 919) |
| `--results_dir` | `results/` | Directory for CSV outputs |
| `--fig_dir` | `figures/` | Directory for figures |
| `--top_k_heads` | `10` | How many inhibition heads to select for amplification |
| `--amp_scales` | `0.5 1 2 3 4` | Amplification scale values to sweep |
| `--skip_per_head` | off | Skip Stage 5 (per-head decomposition) |
| `--skip_amplification` | off | Skip Stage 6 (amplification experiment) |

### Single-prompt prototype

```bash
python transformerLenstest.py
```

Runs all 7 analysis phases on one hardcoded prompt pair and saves figures to `figures/`.

---

## Single-Prompt Pipeline (`transformerLenstest.py`)

Runs all analysis phases on one hardcoded prompt pair:

- **Positive**: `"The capital of France is"` → target: `" Paris"`
- **Negated**: `"The capital of France is not"` → should NOT predict `" Paris"`

### Phase 0 — Setup
Loads GPT-2 Small via TransformerLens and defines the prompt pair.

### Phase 1 — Behavioural Comparison
Checks surface-level output: where does `" Paris"` rank for each prompt? A high rank on the negated prompt confirms the **negation failure**.

### Phase 2 — Residual Stream Decomposition
Uses `cache.decompose_resid()` to break the final residual stream into each component's (embedding, attention, FFN) individual contribution — essentially "who wrote what" into the model's working memory.

### Phase 3 — Direct Logit Attribution (DLA) 
The heart of the project. Since the residual stream is a sum of all component outputs, the final logit for any token $t$ can be attributed component-by-component:

$$\text{logit}(t) = \sum_{c} \mathbf{r}_c \cdot \mathbf{W}_U[:, t]$$

- **Positive DLA** → component pushes the model *toward* `" Paris"`
- **Negative DLA** → component pushes the model *away from* `" Paris"`

### Phase 4 — Memory vs Inhibition Separation
Components are split into two camps:
- **FFN layers** = Memory/Retrieval (Geva et al.'s "key-value stores")
- **Attention layers** = Logic/Inhibition (Hanna et al.'s "inhibition heads")

**Key hypothesis**: FFN DLA values should be nearly identical between positive and negated prompts (retrieval is logic-blind), while attention DLA should differ dramatically (where negation processing happens, or fails to).

### Phase 5 — Signal-to-Gate Ratio (SGR) Novel Metric

$$\text{SGR} = \frac{|\text{FFN layers pushing toward target}|}{|\text{Attn layers pushing away from target}|}$$

- $\text{SGR} > 1$ → memory overwhelms logic → **hallucination / negation failure**
- $\text{SGR} < 1$ → logic successfully suppresses → **correct behaviour**

The **crossover point** (Phase 5b) plots cumulative DLA layer-by-layer to show whether inhibition ever overcomes retrieval.

### Phase 6 — Activation Patching (Causal Intervention)
For each layer, the negated-prompt activation is swapped with the positive-prompt activation and the change in target logit is measured. Large positive Δ identifies layers causally responsible for negation processing. Performed separately for residual stream, MLP outputs, and attention outputs.

### Phase 7 — Visualization
Saves 5 publication-quality figures to `figures/`:

| File | Content |
|---|---|
| `01_ffn_dla_comparison.png` | FFN DLA: positive vs negated |
| `02_attn_dla_comparison.png` | Attention DLA: positive vs negated |
| `03_cumulative_dla_crossover.png` | Cumulative DLA with crossover point |
| `04_activation_patching.png` | Activation patching results (3-panel) |
| `05_sgr.png` | SGR bar chart |

---

## Full Dataset Pipeline (`run_pipeline.py`)

Runs all stages over the complete CounterFact dataset on one or more models.

### Stage 1 — Model Loading
Loads the specified model(s) via TransformerLens with `use_attn_result=True` (required for per-head hooks), LayerNorm folding, and centred weights for exact DLA.

### Stage 2 — Dataset Loading
Loads [NeelNanda/counterfact-tracing](https://huggingface.co/datasets/NeelNanda/counterfact-tracing) from HuggingFace. Builds `PromptPair` objects (positive + negated prompt, target token). Skips entries where the target is not a single token.

### Stage 3 — Benchmark (DLA + SGR per prompt pair)
Runs the full DLA + SGR analysis on every prompt pair and saves results to `results/{model}_benchmark.csv` (one row per pair, including per-layer DLA arrays).

### Stage 4 — SGR Distribution Analysis
Analyses the Signal-to-Gate Ratio distribution across the dataset.

$$\text{SGR} = \frac{|\text{FFN DLA pushing toward target}|}{|\text{Attn DLA pushing away from target}|}$$

- $\text{SGR} > 1$ → memory overwhelms inhibition → **negation failure**
- $\text{SGR} < 1$ → inhibition overrides memory → **correct suppression**

Produces: `sgr_histogram.png`, `sgr_failure_rate.png`, `per_layer_dla_mean.png`

### Stage 5 — Per-Head Decomposition
Identifies specific **inhibition heads** — attention heads whose DLA drops most dramatically from the positive to the negated prompt — by computing mean ΔDLA across the dataset.

Produces: `head_dla_heatmap.png`

### Stage 6 — Artificial Amplification
Scales the outputs of the top inhibition heads by 0.5×, 2×, 3×, … and measures whether this reduces the negation failure rate across the dataset.

Produces: `amplification_sweep.png`, `amplification_failure_rate.png`

### Stage 7 — Statistical Analysis
- Spearman / point-biserial correlation of SGR with negation failure flag
- Mann-Whitney U test comparing SGR distributions of failures vs successes
- Bootstrap 95% confidence intervals on failure rates
- Cross-model significance test (two-proportion z-test + Mann-Whitney on SGR)

---

## Key Metrics

| Metric | Definition |
|---|---|
| **Negation Failure Rate** | Fraction of prompts where the model still top-ranks the forbidden target after negation |
| **SGR** | FFN retrieval signal / attention inhibition signal; SGR > 1 predicts failure |
| **ΔDLA (per head)** | Drop in head DLA from positive → negated prompt; large Δ = inhibition head |
| **Crossover Layer** | Layer where cumulative inhibition first exceeds cumulative retrieval (or never does) |

---

## Reproducibility

Fixed seed throughout:

```python
torch.manual_seed(67)
```