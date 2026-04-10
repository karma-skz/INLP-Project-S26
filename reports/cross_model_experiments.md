# Cross-Model Experiment Report

- Models rerun: `gpt2-small`, `gpt2-medium`, `gpt2-large`, `pythia-160m`, `pythia-410m`
- Negator suffix: ` not`
- Benchmark samples per model: `200`
- Per-model benchmark CSVs: `results/cross_model`
- Combined benchmark CSV: `results/cross_model/all_models_benchmark.csv`

## Benchmark Summary

| model_name | n_samples | n_failures | failure_rate | median_rank_shift | median_sgr | success_median_sgr | failure_median_sgr |
|---|---|---|---|---|---|---|---|
| gpt2-medium | 200 | 26 | 0.1300 | 247.5000 | 67.0839 | 66.8672 | 68.4960 |
| gpt2-large | 200 | 25 | 0.1250 | 73.0000 | 56.8514 | 56.6070 | 60.5880 |
| gpt2-small | 200 | 24 | 0.1200 | 707.0000 | 29.2806 | 27.2789 | 38.3634 |
| pythia-160m | 200 | 22 | 0.1100 | 900.5000 | 24.1762 | 24.5675 | 22.2137 |
| pythia-410m | 200 | 19 | 0.0950 | 258.0000 | 6.0129 | 5.9134 | 6.6541 |

Interpretation:

- `median_rank_shift` is `neg_target_rank - pos_target_rank`; positive values mean negation usually pushes the factual token downward.
- `success_median_sgr` and `failure_median_sgr` show whether the SGR ordering matches the behavioural split.

## SGR Edge Cases

| model_name | success_with_sgr_gt1 | failure_with_sgr_le1 | success_mismatch_rate | failure_mismatch_rate |
|---|---|---|---|---|
| gpt2-large | 148 | 0 | 0.8457 | 0.0000 |
| gpt2-medium | 127 | 0 | 0.7299 | 0.0000 |
| gpt2-small | 110 | 0 | 0.6250 | 0.0000 |
| pythia-160m | 170 | 0 | 0.9551 | 0.0000 |
| pythia-410m | 181 | 0 | 1.0000 | 0.0000 |

Interpretation:

- `success_with_sgr_gt1` counts cases where the model suppresses the target even though SGR is above 1.
- `failure_with_sgr_le1` counts cases that go against the simple SGR-threshold story.

## Head Selection and Interventions

| model_name | analysis_pairs | top_heads |
|---|---|---|
| gpt2-large | 100 | (25,13), (30,6), (26,17), (32,11), (19,0), ... |
| gpt2-medium | 100 | (21,12), (20,6), (16,2), (15,0), (20,12), ... |
| gpt2-small | 100 | (9,8), (10,0), (8,11), (11,2), (0,1), ... |
| pythia-160m | 100 | (9,4), (8,1), (8,10), (7,8), (9,0), ... |
| pythia-410m | 100 | (17,6), (19,10), (17,13), (16,15), (19,9), ... |

| model_name | analysis_pairs | baseline_rate | best_rate | best_scale | absolute_improvement |
|---|---|---|---|---|---|
| gpt2-large | 100 | 0.1300 | 0.1200 | 0.5000 | 0.0100 |
| gpt2-medium | 100 | 0.1300 | 0.0900 | 4.0000 | 0.0400 |
| gpt2-small | 100 | 0.1200 | 0.0600 | 4.0000 | 0.0600 |
| pythia-160m | 100 | 0.0800 | 0.0200 | 4.0000 | 0.0600 |
| pythia-410m | 100 | 0.1300 | 0.0800 | 4.0000 | 0.0500 |

| model_name | patch_samples | best_patch_type | best_layer | best_delta | best_resid | best_mlp | best_attn |
|---|---|---|---|---|---|---|---|
| gpt2-large | 10 | resid | 32.0000 | 1.2354 | L32 (+1.235) | L23 (+0.444) | L14 (+0.129) |
| gpt2-medium | 10 | resid | 2.0000 | 2.5467 | L2 (+2.547) | L18 (+0.892) | L21 (+0.177) |
| gpt2-small | 10 | resid | 9.0000 | 2.5497 | L9 (+2.550) | L0 (+1.545) | L9 (+0.175) |
| pythia-160m | 10 | resid | 10.0000 | 3.5547 | L10 (+3.555) | L11 (+1.072) | L8 (+0.213) |
| pythia-410m | 10 | resid | 5.0000 | 2.4114 | L5 (+2.411) | L0 (+1.002) | L1 (+0.536) |

## What Helps Our Story

- `gpt2-medium` shows higher median SGR in failures than successes (68.50 vs 66.87), which supports the retrieval-over-inhibition story.
- `gpt2-large` shows higher median SGR in failures than successes (60.59 vs 56.61), which supports the retrieval-over-inhibition story.
- `gpt2-small` shows higher median SGR in failures than successes (38.36 vs 27.28), which supports the retrieval-over-inhibition story.
- `pythia-410m` shows higher median SGR in failures than successes (6.65 vs 5.91), which supports the retrieval-over-inhibition story.
- `gpt2-large` improves under head amplification: failure rate drops from 13.0% to 12.0% at scale 0.50.
- `gpt2-medium` improves under head amplification: failure rate drops from 13.0% to 9.0% at scale 4.00.
- `gpt2-small` improves under head amplification: failure rate drops from 12.0% to 6.0% at scale 4.00.
- `pythia-160m` improves under head amplification: failure rate drops from 8.0% to 2.0% at scale 4.00.
- `pythia-410m` improves under head amplification: failure rate drops from 13.0% to 8.0% at scale 4.00.
- `gpt2-large` has a positive mean patching effect, strongest for resid at layer 32 (Δ logit +1.235).
- `gpt2-medium` has a positive mean patching effect, strongest for resid at layer 2 (Δ logit +2.547).
- `gpt2-small` has a positive mean patching effect, strongest for resid at layer 9 (Δ logit +2.550).
- `pythia-160m` has a positive mean patching effect, strongest for resid at layer 10 (Δ logit +3.555).
- `pythia-410m` has a positive mean patching effect, strongest for resid at layer 5 (Δ logit +2.411).
- `gpt2-large` rarely produces failure cases with SGR <= 1 (0.0%), so the metric usually points in the right direction for failures.
- `gpt2-medium` rarely produces failure cases with SGR <= 1 (0.0%), so the metric usually points in the right direction for failures.
- `gpt2-small` rarely produces failure cases with SGR <= 1 (0.0%), so the metric usually points in the right direction for failures.
- `pythia-160m` rarely produces failure cases with SGR <= 1 (0.0%), so the metric usually points in the right direction for failures.
- `pythia-410m` rarely produces failure cases with SGR <= 1 (0.0%), so the metric usually points in the right direction for failures.

## What Weakens Our Story

- `gpt2-large` still has many successful suppressions with SGR > 1 (84.6% of successes), so SGR is a useful trend metric, not a clean decision rule.
- `gpt2-medium` still has many successful suppressions with SGR > 1 (73.0% of successes), so SGR is a useful trend metric, not a clean decision rule.
- `gpt2-small` still has many successful suppressions with SGR > 1 (62.5% of successes), so SGR is a useful trend metric, not a clean decision rule.
- `pythia-160m` still has many successful suppressions with SGR > 1 (95.5% of successes), so SGR is a useful trend metric, not a clean decision rule.
- `pythia-410m` still has many successful suppressions with SGR > 1 (100.0% of successes), so SGR is a useful trend metric, not a clean decision rule.
- `gpt2-large` is not fully rescued by amplification; even the best scale leaves a 12.0% failure rate.
- `gpt2-medium` is not fully rescued by amplification; even the best scale leaves a 9.0% failure rate.
- `gpt2-small` is not fully rescued by amplification; even the best scale leaves a 6.0% failure rate.
- `pythia-160m` is not fully rescued by amplification; even the best scale leaves a 2.0% failure rate.
- `pythia-410m` is not fully rescued by amplification; even the best scale leaves a 8.0% failure rate.
- Model differences remain large: `gpt2-medium` fails more often than `pythia-410m` (13.0% vs 9.5%), so the story is not equally strong across architectures.

## Notes

- No graphs are generated by this report. The evidence is intentionally reduced to tables and concise observations.
- The strongest intervention evidence comes from dataset-level head amplification and activation patching, both run on capped subsets for tractability.
- Pair filtering still depends on single-token targets for each tokenizer, so sample counts differ by model.
