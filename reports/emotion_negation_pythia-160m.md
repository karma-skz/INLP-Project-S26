# Emotion Negation Report: pythia-160m

## Setup

- Model: `pythia-160m`
- Representation: `final_token`
- Prompt examples: `192`
- Emotion pairs: `84` affirmed and `84` negated
- Neutral controls: `24`
- Reference layers: `{'early': 3, 'middle': 6, 'late': 11}`

## Peak Direction Summary

| emotion | peak_layer | peak_direction_norm | peak_probe_accuracy | early | late | late_over_early |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 9 | 9.5951 | 1.0000 | 4.1231 | 8.2314 | 1.9964 |
| fear | 9 | 8.7438 | 1.0000 | 4.3917 | 7.3913 | 1.6830 |
| joy | 11 | 10.5321 | 1.0000 | 4.1116 | 8.0105 | 1.9483 |
| sadness | 9 | 8.4006 | 1.0000 | 4.0824 | 7.0885 | 1.7364 |

Interpretation:

- `peak_direction_norm` is the strongest affirmed-vs-negated separation for that emotion.
- `late_over_early` compares mean late-layer strength to early-layer strength; values above 1 mean the direction gets stronger deeper in the network.

## Negation Behaviour At The Peak Layer

| emotion | peak_layer | opposite_emotion | distance_to_neutral | distance_to_opposite | distance_gap | closer_to_neutral_rate |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 9 |  | 15.0387 |  |  |  |
| fear | 9 |  | 14.8869 |  |  |  |
| joy | 11 | sadness | 20.0648 | 14.9678 | -5.0970 | 0.2222 |
| sadness | 9 | joy | 13.3176 | 14.2071 | 0.8895 | 0.7778 |

Interpretation:

- Negative `distance_gap` means the negated representation is closer to the opposite emotion than to neutral.
- `closer_to_neutral_rate` above 0.5 means negation behaves more like attenuation than a polarity flip for most prompts.

## Peak-Layer Direction Injection

| emotion | peak_layer | contrast_label | slope | linearity_r2 | start_margin | end_margin |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 9 | neutral | -0.0000 | 0.2866 | -0.0001 | -0.0004 |
| fear | 9 | neutral | -0.0002 | 0.8718 | 0.0006 | -0.0002 |
| joy | 11 | sadness | 0.0004 | 0.9361 | -0.0005 | 0.0014 |
| sadness | 9 | joy | 0.0003 | 0.9635 | -0.0010 | 0.0007 |

Interpretation:

- Positive `slope` means moving along the learned direction increases target emotion mass relative to the contrast set.
- Higher `linearity_r2` means the effect changes more predictably as the intervention strength grows.

## What Helps The Story

- All emotions peak in the middle-to-late layers, which supports the claim that negation-sensitive directions consolidate late.
- Peak-layer probe accuracy stays high across emotions, so affirmed and negated prompts are linearly separable once the representation is formed.
- At least some emotions move closer to neutral than to their opposite under negation, which helps the attenuation-not-flip interpretation.
- Some peak-layer direction injections remain reasonably linear, so the learned directions are not purely descriptive.

## What Hurts The Story

- `joy` do not reliably move toward neutral at the peak layer, which weakens a universal attenuation claim.
- `anger` show weak linearity at the peak layer, so the causal direction story is uneven across emotions.
- Some emotions still lack explicit opposite-emotion controls, so the neutral-vs-opposite conclusion is incomplete.

## Bottom Line

- This report is text-only by design: no PCA, heatmaps, or line plots are used in the write-up.
- The most useful pieces here are where the direction peaks, whether it strengthens late, and whether negation moves emotions toward neutral or somewhere else.
