# Cross-Model Experiment Report

- Models rerun: `gpt2-small`, `pythia-160m`
- Negator suffix: ` not`
- Benchmark samples per model: `all available`
- Per-head/amplification sample cap: `200`
- Activation patching sample cap: `20`
- Combined benchmark CSV: `results/cross_model/all_models_benchmark.csv`

## Benchmark Summary

| model_name | n_samples | failure_rate | sgr_mean | sgr_median | sgr_gt1_rate_finite | crossover_present |
|---|---|---|---|---|---|---|
| gpt2-small | 21919 | 0.1138 | 244.5871 | 27.0643 | 0.9912 | 0.5644 |
| pythia-160m | 19562 | 0.0914 | 212.2123 | 24.4571 | 1.0000 | 0.5507 |

## Correlation Summary

| model_name | n_samples | spearman_r | spearman_p | pointbiserial_r | pointbiserial_p |
|---|---|---|---|---|---|
| gpt2-small | 21919 | 0.1499 | 0.0000 | 0.0044 | 0.6179 |
| pythia-160m | 19562 | 0.0126 | 0.0866 | -0.0034 | 0.6483 |

## Failure Rate Confidence Intervals

| model_name | n_samples | n_failures | failure_rate | ci_lower | ci_upper |
|---|---|---|---|---|---|
| gpt2-small | 21919 | 2495 | 0.1138 | 0.1099 | 0.1179 |
| pythia-160m | 19562 | 1787 | 0.0914 | 0.0873 | 0.0956 |

## Per-Head and Amplification Summary

| model_name | top_heads | head_heatmap |
|---|---|---|
| gpt2-small | (9,8), (10,0), (8,11), (11,2), (0,1) | figures/cross_model/gpt2-small_head_dla_heatmap.png |
| pythia-160m | (9,4), (8,1), (8,10), (7,8), (9,0) | figures/cross_model/pythia-160m_head_dla_heatmap.png |

| model_name | baseline_rate_at_1x | best_rate | best_scale | failure_curve_figure | single_prompt_figure |
|---|---|---|---|---|---|
| gpt2-small | 0.1200 | 0.0600 | 4.0000 | figures/cross_model/gpt2-small_amplification_failure_rate.png | figures/cross_model/gpt2-small_amplification_sweep.png |
| pythia-160m | 0.1100 | 0.0500 | 4.0000 | figures/cross_model/pythia-160m_amplification_failure_rate.png | figures/cross_model/pythia-160m_amplification_sweep.png |

- Cross-model amplification figure: `figures/cross_model/cross_model_amplification_failure_rate.png`

## Activation Patching Summary

| model_name | patch_samples | best_resid | best_mlp | best_attn | patch_figure |
|---|---|---|---|---|---|
| gpt2-small | 20 | L9 (+2.610) | L0 (+1.874) | L3 (+0.172) | figures/cross_model/gpt2-small_activation_patching.png |
| pythia-160m | 20 | L11 (+4.272) | L0 (+1.504) | L8 (+0.315) | figures/cross_model/pythia-160m_activation_patching.png |

- Cross-model activation patching figure: `figures/cross_model/cross_model_activation_patching.png`

## Shared SGR Figures

- Histogram: `figures/cross_model/sgr_histogram.png`
- Failure-rate curve: `figures/cross_model/sgr_failure_rate.png`
- Per-layer DLA heatmap: `figures/cross_model/per_layer_dla_mean.png`
- Model comparison: `figures/cross_model/sgr_model_comparison.png`

## Two-Model Statistical Comparison

- Failure rates: `gpt2-small` = 0.1138, `pythia-160m` = 0.0914
- Two-proportion z-test: `z = 7.5111`, `p = 5.864e-14`
- Mann-Whitney on SGR: `U = 120125434`, `p = 0.02799`

## Notes

- `sgr_gt1_rate_finite` is computed over finite SGR values only, which matches the standalone SGR analysis.
- SGR plots now use a data-aware range instead of a fixed clip; the current combined run used an automatic cap derived from the plotted data.
- Amplification and activation-patching figures also use dynamic y-axis ranges based on the actual plotted values.
