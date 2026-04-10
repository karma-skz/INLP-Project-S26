# INLP Project: Negation Failures in Language Models

This repository studies negation failure in causal language models using mechanistic interpretability tools from TransformerLens.

The central question is simple:

- why does a model still strongly predict a factual token after seeing negation?
- which internal components retrieve that fact?
- which components try to suppress it?
- can we measure and intervene on that competition?

The project analyzes this with:

- Direct Logit Attribution (DLA)
- the Signal-to-Gate Ratio (SGR)
- per-head attention analysis
- activation patching
- inhibition-head amplification

## Models

Supported model names:

- `gpt2-small`
- `gpt2-medium`
- `gpt2-large`
- `pythia-160m`
- `pythia-410m`

## Repository Structure

```text
INLP-Project-S26/
├── run_pipeline.py
├── run_emotion_negation_experiment.py
├── run_cross_model_experiments.py
├── run_final_qualitative_analysis.py
├── figures/
├── reports/
│   ├── final/
│   ├── cross_model_experiments.md
│   └── qualitative.md
└── src/
    ├── analysis/
    ├── benchmark/
    ├── dataset/
    ├── metrics/
    ├── models/
    ├── reporting/
    └── utils/
```

## Setup

Using conda:

```bash
conda env create -f environment.yml
conda activate inlp-project
```

Using pip:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the default pipeline on `gpt2-small` with a small sample:

```bash
python run_pipeline.py
```

This generates:

- benchmark CSVs in `results/`
- figures in `figures/`
- printed summary statistics in the terminal

The most important files after a run are usually:

- `results/gpt2-small_not_benchmark.csv`
- `results/all_models_benchmark.csv`
- `figures/sgr_histogram.png`
- `figures/sgr_failure_rate.png`
- `figures/per_layer_dla_mean.png`
- `figures/head_dla_heatmap.png`
- `figures/amplification_sweep.png`
- `figures/amplification_failure_rate.png`

## Main Experiment Pipeline

Run the main benchmark:

```bash
python run_pipeline.py
```

Run both default models on the full dataset:

```bash
python run_pipeline.py --models gpt2-small pythia-160m --max_samples -1
```

Run the new emotion-negation direction experiment:

```bash
python run_emotion_negation_experiment.py --models gpt2-small
```

Run a faster version that skips the slower head-level and amplification stages:

```bash
python run_pipeline.py --skip_per_head --skip_amplification
```

Useful options:

- `--models` selects one or more models
- `--max_samples -1` uses the full dataset
- `--negator_suffix` lets you test alternative negation strings
- `--results_dir` changes where CSVs are saved
- `--fig_dir` changes where figures are saved

### What the pipeline does

`run_pipeline.py` performs these stages:

1. loads model(s)
2. loads CounterFact prompt pairs
3. computes benchmark metrics including DLA and SGR
4. generates SGR distribution plots
5. identifies top inhibition heads
6. runs amplification experiments
7. prints statistical summaries

### Where to look after running

Open the benchmark CSVs in `results/` to inspect per-example outputs such as:

- `pos_target_rank`
- `neg_target_rank`
- `negation_failure`
- `retrieval_strength`
- `inhibition_strength`
- `sgr`
- `crossover_layer`

Open the figures in `figures/` to inspect aggregate behavior across the run.

## Cross-Model Experiment Report

To run a more complete cross-model comparison and write a report:

```bash
python run_cross_model_experiments.py --models gpt2-small pythia-160m --max_samples -1
```

This writes:

- per-model CSVs to `results/cross_model/`
- cross-model figures to `figures/cross_model/`
- a markdown summary to `reports/cross_model_experiments.md`

Useful outputs from this run include:

- `results/cross_model/all_models_benchmark.csv`
- `figures/cross_model/sgr_model_comparison.png`
- `figures/cross_model/cross_model_amplification_failure_rate.png`
- `figures/cross_model/cross_model_activation_patching.png`
- `reports/cross_model_experiments.md`

## Emotion Negation Direction Experiment

This experiment extends the repository with a controlled emotional-representation benchmark inspired by Anthropic's linear-direction results.

It builds balanced prompt pairs such as:

- `I feel happy`
- `I do not feel happy`

across multiple emotion categories (`joy`, `sadness`, `anger`, `fear`) plus neutral controls.

Run it with:

```bash
python run_emotion_negation_experiment.py --models gpt2-small
```

Useful options:

- `--models` selects one or more models
- `--representation final_token|mean_pool` switches between last-token and pooled residual representations
- `--layers` restricts analysis to specific layers
- `--alpha_values` customizes the direction-injection sweep

This writes per-model outputs to:

- `results/emotion_negation/<model>/emotion_prompt_metadata.csv`
- `results/emotion_negation/<model>/emotion_direction_summary.csv`
- `results/emotion_negation/<model>/emotion_opposite_summary.csv`
- `results/emotion_negation/<model>/emotion_negation_sensitivity.csv`
- `results/emotion_negation/<model>/emotion_linearity_summary.csv`
- `results/emotion_negation/<model>/emotion_direction_vectors.npz`

and figures to:

- `figures/emotion_negation/<model>/emotion_direction_overview.png`
- `figures/emotion_negation/<model>/emotion_direction_symmetry.png`
- `figures/emotion_negation/<model>/emotion_negation_sensitivity.png`
- `figures/emotion_negation/<model>/emotion_direction_pca.png`
- `figures/emotion_negation/<model>/emotion_direction_linearity.png`

The markdown report is written to:

- `reports/emotion_negation_<model>.md`

## Qualitative Reports

To build a detailed qualitative package from existing cross-model benchmark CSVs:

```bash
python run_final_qualitative_analysis.py --models gpt2-small pythia-160m --results_dir results/cross_model --n_cases 10
```

This writes:

- markdown report to `reports/qualitative.md`
- case/global CSV outputs to `results/qualitative/`
- plots to `figures/qualitative/`

These reports are useful if you want to inspect example prompts, token ranks, top predictions, and patching behavior for specific cases instead of only looking at aggregate figures.

## Post-Hoc Analyses

The `src.reporting` modules run additional analyses from existing benchmark outputs.

Run crossover analysis:

```bash
python -m src.reporting.crossover
```

Outputs:

- `figures/crossover_layer_dist.png`
- `figures/crossover_vs_failure.png`
- `figures/crossover_vs_sgr.png`

Run the semantic audit:

```bash
python -m src.reporting.semantic_audit
```

Outputs:

- `figures/audit_relation_failure_rates.png`
- `figures/audit_prompt_structure.png`
- `figures/audit_sgr_by_relation.png`

Run SGR verification:

```bash
python -m src.reporting.sgr_verification
```

Outputs:

- `figures/sgr_lt1_verification.png`
- `figures/sgr_lt1_by_negation.png`

Run dataset-scale activation patching:

```bash
python -m src.reporting.activation_patching
```

Output:

- `figures/activation_patching_rescue_rate.png`

Run the extended amplification comparison:

```bash
python -m src.reporting.extended_amplification
```

Output:

- `figures/extended_amplification.png`

## Final Report

The final written report is in:

- `reports/final/final_report.tex`
- `reports/final/final_report.pdf`

If you want the fastest path through the repo, start with:

1. `python run_pipeline.py`
2. open `results/all_models_benchmark.csv`
3. inspect the figures in `figures/`
4. read `reports/final/final_report.pdf`
