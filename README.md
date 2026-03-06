# INLP Project: Negation Failures in Large Language Models

## Project Overview

This project studies **negation failures in large language models (LLMs)** using mechanistic interpretability tools.

Large language models often fail to correctly handle negation — for example, when prompted with _"The capital of France is not"_, many models still assign high probability to _"Paris"_. This project provides infrastructure to:

1. **Benchmark** negation failure rates across models
2. **Analyze** model internals using mechanistic interpretability (TransformerLens)
3. **Reproduce** experiments with fixed seeds and clean configuration

### Supported Models

- **GPT-2 Small** (`gpt2-small`) — 124M parameters
- **Pythia-160M** (`pythia-160m`) — 160M parameters

### Dataset

- [NeelNanda/counterfact-tracing](https://huggingface.co/datasets/NeelNanda/counterfact-tracing) via HuggingFace Datasets

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   └── experiment_config.yaml
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── load_dataset.py
│   │   └── build_prompts.py
│   ├── models/
│   │   └── load_models.py
│   ├── benchmark/
│   │   ├── run_benchmark.py
│   │   └── metrics.py
│   └── utils/
│       └── io_utils.py
│
├── experiments/
│   └── benchmark_experiment.py
│
├── results/
│   └── benchmark_results.json
│
└── notebooks/
    └── exploratory.ipynb
```

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

## Run Benchmark

```bash
python experiments/benchmark_experiment.py
```

This will:

1. Download the CounterFact dataset from HuggingFace
2. Load GPT-2 Small and Pythia-160M via TransformerLens
3. Run the negation failure benchmark
4. Output results to `results/benchmark_results.json`

### Example Output

```json
{
  "gpt2-small": {
    "positive_accuracy": 0.52,
    "negation_failure_rate": 0.37
  },
  "pythia-160m": {
    "positive_accuracy": 0.47,
    "negation_failure_rate": 0.34
  }
}
```

---

## Configuration

Edit `configs/experiment_config.yaml` to customize experiments:

```yaml
dataset_size: 100
models:
  - gpt2-small
  - pythia-160m
device: auto
batch_size: 1
```

---

## Key Metrics

| Metric | Definition |
|---|---|
| **Positive Accuracy** | Model predicts the correct target token on the factual prompt |
| **Negation Failure Rate** | Model still predicts the factual target even after negation |

---

## Reproducibility

All experiment scripts set fixed seeds for reproducibility:

```python
torch.manual_seed(68)
numpy.random.seed(68)
```

---

## License

This project is for academic research purposes.