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
├── condaenv.ampjwu_m.requirements.txt
├── .gitignore
├── explanation.md              # Pipeline walkthrough and project roadmap
├── transformerLenstest.py      # Full pipeline exploration script
└── figures/                    # Auto-generated output figures
    ├── 01_ffn_dla_comparison.png
    ├── 02_attn_dla_comparison.png
    ├── 03_cumulative_dla_crossover.png
    ├── 04_activation_patching.png
    └── 05_sgr.png
```

---

## Pipeline Overview

`transformerLenstest.py` runs the full analysis pipeline on a single prompt pair:

- **Positive**: `"The capital of France is"` → target: `" Paris"`
- **Negated**: `"The capital of France is not"` → should NOT predict `" Paris"`

### Phase 0 — Setup
Loads GPT-2 Small via TransformerLens and defines the prompt pair.

### Phase 1 — Behavioural Comparison
Checks surface-level output: where does `"Paris"` rank for each prompt? A high rank on the negated prompt confirms the **negation failure**.

### Phase 2 — Residual Stream Decomposition
Uses `cache.decompose_resid()` to break the final residual stream into each component's (embedding, attention, FFN) individual contribution — essentially "who wrote what" into the model's working memory.

### Phase 3 — Direct Logit Attribution (DLA) ⭐
The heart of the project. Since the residual stream is a sum of all component outputs, the final logit for any token $t$ can be attributed component-by-component:

$$\text{logit}(t) = \sum_{c} \mathbf{r}_c \cdot \mathbf{W}_U[:, t]$$

- **Positive DLA** → component pushes the model *toward* `" Paris"`
- **Negative DLA** → component pushes the model *away from* `" Paris"`

### Phase 4 — Memory vs Inhibition Separation
Components are split into two camps:
- **FFN layers** = Memory/Retrieval (Geva et al.'s "key-value stores")
- **Attention layers** = Logic/Inhibition (Hanna et al.'s "inhibition heads")

**Key hypothesis**: FFN DLA values should be nearly identical between positive and negated prompts (retrieval is logic-blind), while attention DLA should differ dramatically (where negation processing happens, or fails to).

### Phase 5 — Signal-to-Gate Ratio (SGR) ⭐ Novel Metric

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

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/karma-skz/INLP-Project-S26.git
cd INLP-Project-S26
```

### 2. Create the conda environment

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

```bash
python transformerLenstest.py
```

This will load GPT-2 Small, run all 7 phases on the example prompt pair, print phase-by-phase outputs to the terminal, and save figures to `figures/`.

---

## Project Roadmap

| Step | What | How |
|---|---|---|
| **1** | Run single-prompt exploration | `python transformerLenstest.py` |
| **2** | Scale to full dataset | Load CounterFact, loop through all entries |
| **3** | Compute SGR distribution | Plot SGR histogram across prompt pairs → `src/benchmark/sgr_analysis.py` |
| **4** | Per-head decomposition | Use `cache.decompose_resid(mode="attn")` at head level to find specific inhibition heads |
| **5** | Add Pythia-160m | Change `model_name` to `"pythia-160m"` in model loader |
| **6** | Artificial amplification | Hook inhibition heads and multiply outputs by 2×/3× to test if negation can be "fixed" |
| **7** | Statistical analysis | Correlate SGR with negation failure rate, compute p-values → `metrics.py` |

---

## Key Metrics

| Metric | Definition |
|---|---|
| **Positive Accuracy** | Model predicts the correct target token on the factual prompt |
| **Negation Failure Rate** | Model still predicts the factual target after negation |
| **SGR** | Ratio of memory signal to inhibition gate; SGR > 1 indicates failure |

---

## Reproducibility

The pipeline uses a fixed seed for reproducibility:

```python
torch.manual_seed(67)
```