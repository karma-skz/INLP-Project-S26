# Emotion Negation Report: gpt2-large

## Setup

- Model: `gpt2-large`
- Representation: `final_token`
- Prompt examples: `216`
- Emotion pairs: `96` affirmed and `96` negated
- Neutral controls: `24`
- Reference layers: `{'early': 9, 'middle': 18, 'late': 35}`

## Peak Direction Summary

| emotion | peak_layer | peak_direction_norm | peak_probe_accuracy | early | late | late_over_early |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 35 | 122.2576 | 1.0000 | 20.0407 | 79.7881 | 3.9813 |
| fear | 35 | 110.3076 | 1.0000 | 21.9380 | 75.4614 | 3.4398 |
| joy | 35 | 102.8842 | 1.0000 | 17.8566 | 68.6411 | 3.8440 |
| sadness | 35 | 106.8202 | 1.0000 | 19.6012 | 73.6127 | 3.7555 |

Interpretation:

- `peak_direction_norm` is the strongest affirmed-vs-negated separation for that emotion.
- `late_over_early` compares mean late-layer strength to early-layer strength; values above 1 mean the direction gets stronger deeper in the network.

## Negation Behaviour At The Peak Layer

| emotion | peak_layer | opposite_emotion | distance_to_neutral | distance_to_opposite | distance_gap | closer_to_neutral_rate |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 35 |  | 194.3113 |  |  |  |
| fear | 35 |  | 195.0575 |  |  |  |
| joy | 35 | sadness | 172.9501 | 158.6382 | -14.3119 | 0.2083 |
| sadness | 35 | joy | 178.7723 | 163.9963 | -14.7759 | 0.0417 |

Interpretation:

- Negative `distance_gap` means the negated representation is closer to the opposite emotion than to neutral.
- `closer_to_neutral_rate` above 0.5 means negation behaves more like attenuation than a polarity flip for most prompts.

## Peak-Layer Direction Injection

| emotion | peak_layer | contrast_label | slope | linearity_r2 | start_margin | end_margin |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 35 | neutral | -0.0000 | 0.6383 | 0.0002 | 0.0001 |
| fear | 35 | neutral | -0.0001 | 0.8577 | 0.0003 | 0.0000 |
| joy | 35 | sadness | -0.0003 | 0.8370 | -0.0008 | -0.0026 |
| sadness | 35 | joy | 0.0009 | 0.9785 | 0.0002 | 0.0048 |

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
