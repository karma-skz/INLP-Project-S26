# Cross-Model Experiment Report

- Models rerun: `gpt2-small`, `pythia-160m`
- Negator suffix: ` not`
- Benchmark samples per model: `all`
- Per-model benchmark CSVs: `results/cross_model`
- Combined benchmark CSV: `results/cross_model/all_models_benchmark.csv`

## Benchmark Summary Metrics

| model_name | n_samples | n_failures | failure_rate | median_rank_shift | median_sgr | success_median_sgr | failure_median_sgr |
|---|---|---|---|---|---|---|---|
| gpt2-small | 21919 | 2498 | 0.1140 | 668.0000 | 27.0919 | 24.8321 | 48.1335 |
| pythia-160m | 19562 | 1764 | 0.0902 | 938.5000 | 24.4801 | 24.4812 | 24.4402 |

## Outcome Metrics

| model_name | outcome | n_samples | median_rank_shift | mean_rank_shift | median_sgr | mean_sgr |
|---|---|---|---|---|---|---|
| gpt2-small | failure | 2498 | -174.5000 | -1005.117293835068 | 48.1335 | 225.2213895598842 |
| gpt2-small | success | 19421 | 957.0000 | 3012.221461304773 | 24.8321 | 184.46794479893128 |
| pythia-160m | failure | 1764 | -296.5000 | -1278.469954648526 | 24.4402 | 47.76795863191455 |
| pythia-160m | success | 17798 | 1198.0000 | 2998.460332621643 | 24.4812 | 128.1950598068973 |

## SGR Edge Case Metrics

| model_name | success_with_sgr_gt1 | failure_with_sgr_le1 | success_mismatch_rate | failure_mismatch_rate |
|---|---|---|---|---|
| gpt2-small | 10893 | 0 | 0.5609 | 0.0000 |
| pythia-160m | 16724 | 0 | 0.9397 | 0.0000 |

## Head Selection Metrics

| model_name | analysis_pairs | top_heads |
|---|---|---|
| gpt2-small | 1000 | (9,8), (10,0), (8,11), (0,1), (11,2), ... |
| pythia-160m | 1000 | (9,4), (8,1), (8,10), (7,8), (9,0), ... |

## Amplification Metrics

| model_name | analysis_pairs | baseline_rate | best_rate | best_scale | absolute_improvement |
|---|---|---|---|---|---|
| gpt2-small | 1000 | 0.1060 | 0.0550 | 4.0000 | 0.0510 |
| pythia-160m | 1000 | 0.0780 | 0.0340 | 4.0000 | 0.0440 |

## Activation Patching Metrics

| model_name | patch_samples | best_patch_type | best_layer | best_delta | best_resid | best_mlp | best_attn |
|---|---|---|---|---|---|---|---|
| gpt2-small | 1000 | resid | 9.0000 | 3.6906 | L9 (+3.691) | L0 (+2.904) | L2 (+0.217) |
| pythia-160m | 1000 | resid | 3.0000 | 5.3380 | L3 (+5.338) | L0 (+2.684) | L8 (+0.232) |
