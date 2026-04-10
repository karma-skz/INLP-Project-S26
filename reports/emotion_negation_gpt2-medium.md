# Emotion Negation Report: gpt2-medium

## Setup

- Model: `gpt2-medium`
- Representation: `final_token`
- Prompt examples: `216`
- Emotion pairs: `96` affirmed and `96` negated
- Neutral controls: `24`
- Reference layers: `{'early': 6, 'middle': 12, 'late': 23}`

## Peak Direction Summary

| emotion | peak_layer | peak_direction_norm | peak_probe_accuracy | early | late | late_over_early |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 23 | 140.6130 | 1.0000 | 27.2492 | 94.5095 | 3.4683 |
| fear | 23 | 138.9653 | 1.0000 | 29.3651 | 93.5536 | 3.1859 |
| joy | 23 | 138.4543 | 1.0000 | 26.1201 | 90.2621 | 3.4557 |
| sadness | 23 | 143.1564 | 1.0000 | 26.6014 | 89.0505 | 3.3476 |

Interpretation:

- `peak_direction_norm` is the strongest affirmed-vs-negated separation for that emotion.
- `late_over_early` compares mean late-layer strength to early-layer strength; values above 1 mean the direction gets stronger deeper in the network.

## Negation Behaviour At The Peak Layer

| emotion | peak_layer | opposite_emotion | distance_to_neutral | distance_to_opposite | distance_gap | closer_to_neutral_rate |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 23 |  | 248.5468 |  |  |  |
| fear | 23 |  | 261.5607 |  |  |  |
| joy | 23 | sadness | 249.3450 | 215.0385 | -34.3065 | 0.0417 |
| sadness | 23 | joy | 249.0312 | 223.9167 | -25.1146 | 0.0417 |

Interpretation:

- Negative `distance_gap` means the negated representation is closer to the opposite emotion than to neutral.
- `closer_to_neutral_rate` above 0.5 means negation behaves more like attenuation than a polarity flip for most prompts.

## Peak-Layer Direction Injection

| emotion | peak_layer | contrast_label | slope | linearity_r2 | start_margin | end_margin |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 23 | neutral | -0.0001 | 0.7213 | 0.0001 | -0.0006 |
| fear | 23 | neutral | -0.0001 | 0.7950 | 0.0001 | -0.0004 |
| joy | 23 | sadness | 0.0004 | 0.6527 | -0.0003 | 0.0022 |
| sadness | 23 | joy | 0.0003 | 0.9096 | -0.0004 | 0.0008 |

Interpretation:

- Positive `slope` means moving along the learned direction increases target emotion mass relative to the contrast set.
- Higher `linearity_r2` means the effect changes more predictably as the intervention strength grows.

## What Helps The Story

- All emotions peak in the middle-to-late layers, which supports the claim that negation-sensitive directions consolidate late.
- Peak-layer probe accuracy stays high across emotions, so affirmed and negated prompts are linearly separable once the representation is formed.
- Some peak-layer direction injections remain reasonably linear, so the learned directions are not purely descriptive.

## What Hurts The Story

- `joy`, `sadness` do not reliably move toward neutral at the peak layer, which weakens a universal attenuation claim.
- Some emotions still lack explicit opposite-emotion controls, so the neutral-vs-opposite conclusion is incomplete.

## Bottom Line

- This report is text-only by design: no PCA, heatmaps, or line plots are used in the write-up.
- The most useful pieces here are where the direction peaks, whether it strengthens late, and whether negation moves emotions toward neutral or somewhere else.
