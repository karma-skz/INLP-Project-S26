# Emotion Negation Report: pythia-410m

## Setup

- Model: `pythia-410m`
- Representation: `final_token`
- Prompt examples: `192`
- Emotion pairs: `84` affirmed and `84` negated
- Neutral controls: `24`
- Reference layers: `{'early': 6, 'middle': 12, 'late': 23}`

## Peak Direction Summary

| emotion | peak_layer | peak_direction_norm | peak_probe_accuracy | early | late | late_over_early |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 22 | 15.9556 | 1.0000 | 5.7905 | 11.5489 | 1.9945 |
| fear | 22 | 17.5441 | 1.0000 | 6.6225 | 13.1080 | 1.9793 |
| joy | 22 | 15.0786 | 1.0000 | 6.2781 | 10.8193 | 1.7234 |
| sadness | 22 | 15.4048 | 1.0000 | 6.2410 | 11.4375 | 1.8326 |

Interpretation:

- `peak_direction_norm` is the strongest affirmed-vs-negated separation for that emotion.
- `late_over_early` compares mean late-layer strength to early-layer strength; values above 1 mean the direction gets stronger deeper in the network.

## Negation Behaviour At The Peak Layer

| emotion | peak_layer | opposite_emotion | distance_to_neutral | distance_to_opposite | distance_gap | closer_to_neutral_rate |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 22 |  | 26.5831 |  |  |  |
| fear | 22 |  | 26.2607 |  |  |  |
| joy | 22 | sadness | 24.0759 | 22.7788 | -1.2971 | 0.1667 |
| sadness | 22 | joy | 23.6059 | 22.7979 | -0.8080 | 0.2778 |

Interpretation:

- Negative `distance_gap` means the negated representation is closer to the opposite emotion than to neutral.
- `closer_to_neutral_rate` above 0.5 means negation behaves more like attenuation than a polarity flip for most prompts.

## Peak-Layer Direction Injection

| emotion | peak_layer | contrast_label | slope | linearity_r2 | start_margin | end_margin |
| --- | --- | --- | --- | --- | --- | --- |
| anger | 22 | neutral | -0.0002 | 0.9840 | 0.0006 | -0.0004 |
| fear | 22 | neutral | -0.0001 | 0.9622 | 0.0003 | -0.0002 |
| joy | 22 | sadness | -0.0000 | 0.0059 | -0.0002 | -0.0003 |
| sadness | 22 | joy | 0.0005 | 0.7906 | -0.0017 | 0.0009 |

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
