# Emotion Negation Report: gpt2-small

## Setup

- Model: `gpt2-small`
- Representation: `final_token`
- Prompt examples: `216`
- Emotion pairs: `96` affirmed and `96` negated
- Neutral controls: `24`
- Reference layers: `{'early': 3, 'middle': 6, 'late': 11}`

## Peak Direction Summary

| emotion | peak_layer | peak_direction_norm | peak_probe_accuracy | early | late | late_over_early |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 11 | 76.4692 | 1.0000 | 13.0253 | 50.2101 | 3.8548 |
| fear | 11 | 70.6258 | 1.0000 | 15.7116 | 49.3328 | 3.1399 |
| joy | 11 | 79.2378 | 1.0000 | 13.4672 | 47.0196 | 3.4914 |
| sadness | 11 | 73.7039 | 1.0000 | 13.7250 | 48.8706 | 3.5607 |

Interpretation:

- `peak_direction_norm` is the strongest affirmed-vs-negated separation for that emotion.
- `late_over_early` compares mean late-layer strength to early-layer strength; values above 1 mean the direction gets stronger deeper in the network.

## Negation Behaviour At The Peak Layer

| emotion | peak_layer | opposite_emotion | distance_to_neutral | distance_to_opposite | distance_gap | closer_to_neutral_rate |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 11 |  | 140.6848 |  |  |  |
| fear | 11 |  | 127.7173 |  |  |  |
| joy | 11 | sadness | 127.5694 | 106.9243 | -20.6451 | 0.0000 |
| sadness | 11 | joy | 127.7672 | 120.2697 | -7.4975 | 0.2083 |

Interpretation:

- Negative `distance_gap` means the negated representation is closer to the opposite emotion than to neutral.
- `closer_to_neutral_rate` above 0.5 means negation behaves more like attenuation than a polarity flip for most prompts.

## Peak-Layer Direction Injection

| emotion | peak_layer | contrast_label | slope | linearity_r2 | start_margin | end_margin |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 11 | neutral | -0.0001 | 0.9783 | 0.0002 | -0.0005 |
| fear | 11 | neutral | -0.0001 | 0.9790 | 0.0004 | -0.0004 |
| joy | 11 | sadness | 0.0000 | 0.2242 | -0.0004 | -0.0002 |
| sadness | 11 | joy | 0.0007 | 0.9017 | -0.0023 | 0.0015 |

Interpretation:

- Positive `slope` means moving along the learned direction increases target emotion mass relative to the contrast set.
- Higher `linearity_r2` means the effect changes more predictably as the intervention strength grows.

## What Helps The Story

- All emotions peak in the middle-to-late layers, which supports the claim that negation-sensitive directions consolidate late.
- Peak-layer probe accuracy stays high across emotions, so affirmed and negated prompts are linearly separable once the representation is formed.
- Some peak-layer direction injections remain reasonably linear, so the learned directions are not purely descriptive.

## What Hurts The Story

- `joy`, `sadness` do not reliably move toward neutral at the peak layer, which weakens a universal attenuation claim.
- `joy` show weak linearity at the peak layer, so the causal direction story is uneven across emotions.
- Some emotions still lack explicit opposite-emotion controls, so the neutral-vs-opposite conclusion is incomplete.

## Bottom Line

- This report is text-only by design: no PCA, heatmaps, or line plots are used in the write-up.
- The most useful pieces here are where the direction peaks, whether it strengthens late, and whether negation moves emotions toward neutral or somewhere else.
