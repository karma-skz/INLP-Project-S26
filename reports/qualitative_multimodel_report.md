# Multi-Model Qualitative Report

- Models: `gpt2-small`, `gpt2-medium`, `gpt2-large`, `pythia-160m`, `pythia-410m`
- Shared case IDs: 180, 57, 65

## Case 180:  Giovanni Battista Riccioli

- Positive prompt: `The domain of activity of Giovanni Battista Riccioli is`
- Negated prompt: `The domain of activity of Giovanni Battista Riccioli is not`
- Target token: ` astronomy`

### gpt2-small

- Outcome: `failure`
- SGR: `8.475`
- Best patch: `attn` at layer `6` with Δ logit `+0.203`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 1.774 | 0.0000 | 14370 |
| Negated | 3.191 | 0.0000 | 8908 |
| Negated (best patch) | 3.395 | 0.0000 | 7661 |

- Negated top predictions: ` the` (0.0638) | ` known` (0.0572) | ` a` (0.0456) | ` only` (0.0421) | ` just` (0.0244)
- Patched top predictions: ` the` (0.0704) | ` only` (0.0531) | ` known` (0.0507) | ` a` (0.0473) | ` just` (0.0310)

### gpt2-medium

- Outcome: `failure`
- SGR: `10.560`
- Best patch: `mlp` at layer `21` with Δ logit `+0.312`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 2.596 | 0.0000 | 10293 |
| Negated | 2.834 | 0.0000 | 8785 |
| Negated (best patch) | 3.146 | 0.0000 | 7495 |

- Negated top predictions: ` known` (0.0994) | ` only` (0.0594) | ` a` (0.0441) | ` the` (0.0346) | ` yet` (0.0247)
- Patched top predictions: ` known` (0.1064) | ` only` (0.0570) | ` a` (0.0444) | ` the` (0.0395) | ` yet` (0.0237)

### gpt2-large

- Outcome: `failure`
- SGR: `6.414`
- Best patch: `mlp` at layer `25` with Δ logit `+0.450`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 1.083 | 0.0000 | 17381 |
| Negated | 1.495 | 0.0000 | 13871 |
| Negated (best patch) | 1.945 | 0.0000 | 11517 |

- Negated top predictions: ` yet` (0.1809) | ` known` (0.0809) | ` well` (0.0494) | ` a` (0.0332) | ` only` (0.0309)
- Patched top predictions: ` yet` (0.1539) | ` known` (0.0934) | ` well` (0.0438) | ` a` (0.0417) | ` the` (0.0292)

### pythia-160m

- Outcome: `success`
- SGR: `52.886`
- Best patch: `mlp` at layer `4` with Δ logit `+0.797`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 16.594 | 0.0000 | 24470 |
| Negated | 16.875 | 0.0000 | 25022 |
| Negated (best patch) | 17.672 | 0.0000 | 21604 |

- Negated top predictions: ` known` (0.1840) | ` yet` (0.0925) | ` a` (0.0437) | ` only` (0.0368) | ` well` (0.0340)
- Patched top predictions: ` known` (0.2051) | ` a` (0.0465) | ` yet` (0.0444) | ` well` (0.0430) | ` defined` (0.0278)

### pythia-410m

- Outcome: `success`
- SGR: `4.935`
- Best patch: `mlp` at layer `19` with Δ logit `+0.865`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 3.412 | 0.0000 | 6134 |
| Negated | 3.334 | 0.0000 | 6407 |
| Negated (best patch) | 4.199 | 0.0000 | 4495 |

- Negated top predictions: ` limited` (0.0682) | ` well` (0.0465) | ` known` (0.0434) | ` a` (0.0411) | ` yet` (0.0383)
- Patched top predictions: ` limited` (0.0699) | ` well` (0.0593) | ` a` (0.0404) | ` the` (0.0349) | ` known` (0.0346)

## Case 57: Gilles Grimandi

- Positive prompt: `Gilles Grimandi was born in`
- Negated prompt: `Gilles Grimandi was born in not`
- Target token: ` Gap`

### gpt2-small

- Outcome: `success`
- SGR: `0.967`
- Best patch: `resid` at layer `10` with Δ logit `+7.695`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 6.164 | 0.0000 | 3021 |
| Negated | -1.527 | 0.0000 | 35724 |
| Negated (best patch) | 6.168 | 0.0000 | 3030 |

- Negated top predictions: ` too` (0.5024) | ` far` (0.1848) | ` much` (0.0712) | ` so` (0.0619) | `-` (0.0367)
- Patched top predictions: ` Paris` (0.0564) | ` the` (0.0446) | ` France` (0.0157) | ` New` (0.0132) | ` Rome` (0.0116)

### gpt2-medium

- Outcome: `success`
- SGR: `2.740`
- Best patch: `resid` at layer `2` with Δ logit `+6.946`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 5.691 | 0.0000 | 2984 |
| Negated | -1.126 | 0.0000 | 33624 |
| Negated (best patch) | 5.820 | 0.0000 | 2860 |

- Negated top predictions: ` far` (0.3210) | ` too` (0.1227) | `-` (0.0775) | ` one` (0.0617) | ` a` (0.0589)
- Patched top predictions: ` Paris` (0.0750) | ` the` (0.0431) | ` France` (0.0255) | ` B` (0.0142) | ` T` (0.0135)

### gpt2-large

- Outcome: `success`
- SGR: `9.264`
- Best patch: `resid` at layer `1` with Δ logit `+6.034`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 4.953 | 0.0000 | 3187 |
| Negated | -0.792 | 0.0000 | 30482 |
| Negated (best patch) | 5.242 | 0.0000 | 2798 |

- Negated top predictions: ` far` (0.2346) | `hern` (0.1368) | `-` (0.1327) | ` one` (0.0417) | ` too` (0.0330)
- Patched top predictions: ` Paris` (0.1609) | ` the` (0.0518) | ` France` (0.0291) | ` Montreal` (0.0144) | ` B` (0.0123)

### pythia-160m

- Outcome: `success`
- SGR: `13.416`
- Best patch: `resid` at layer `10` with Δ logit `+5.250`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 20.594 | 0.0000 | 4784 |
| Negated | 15.344 | 0.0000 | 13998 |
| Negated (best patch) | 20.594 | 0.0000 | 4784 |

- Negated top predictions: ` too` (0.3748) | ` far` (0.2786) | ` long` (0.0308) | `tingham` (0.0298) | `-` (0.0229)
- Patched top predictions: ` the` (0.0545) | ` 18` (0.0335) | ` France` (0.0188) | ` 17` (0.0142) | ` New` (0.0140)

### pythia-410m

- Outcome: `success`
- SGR: `2.210`
- Best patch: `resid` at layer `6` with Δ logit `+7.803`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 7.574 | 0.0000 | 1133 |
| Negated | 0.061 | 0.0000 | 24180 |
| Negated (best patch) | 7.863 | 0.0001 | 989 |

- Negated top predictions: ` far` (0.1918) | ` too` (0.1248) | `-` (0.1076) | ` long` (0.0643) | ` very` (0.0449)
- Patched top predictions: ` Paris` (0.0745) | ` the` (0.0637) | ` France` (0.0268) | ` a` (0.0154) | ` 1944` (0.0126)

## Case 65: Mayer Carl von Rothschild

- Positive prompt: `Mayer Carl von Rothschild found employment in`
- Negated prompt: `Mayer Carl von Rothschild found employment in not`
- Target token: ` Frankfurt`

### gpt2-small

- Outcome: `success`
- SGR: `2.010`
- Best patch: `resid` at layer `1` with Δ logit `+9.814`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 12.477 | 0.0110 | 11 |
| Negated | 2.896 | 0.0000 | 6930 |
| Negated (best patch) | 12.711 | 0.0135 | 11 |

- Negated top predictions: ` only` (0.4546) | ` just` (0.1698) | ` too` (0.0506) | `-` (0.0486) | ` one` (0.0479)
- Patched top predictions: ` the` (0.1769) | ` Germany` (0.0755) | ` London` (0.0454) | ` a` (0.0385) | ` France` (0.0259)

### gpt2-medium

- Outcome: `success`
- SGR: `7314.642`
- Best patch: `resid` at layer `2` with Δ logit `+7.299`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 10.188 | 0.0019 | 38 |
| Negated | 3.170 | 0.0000 | 4676 |
| Negated (best patch) | 10.469 | 0.0023 | 36 |

- Negated top predictions: ` one` (0.3699) | ` only` (0.2708) | ` just` (0.1949) | `-` (0.0226) | ` a` (0.0210)
- Patched top predictions: ` the` (0.3540) | ` a` (0.0615) | ` London` (0.0322) | ` New` (0.0273) | ` Germany` (0.0154)

### gpt2-large

- Outcome: `success`
- SGR: `16.698`
- Best patch: `resid` at layer `1` with Δ logit `+6.168`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 10.906 | 0.0039 | 21 |
| Negated | 5.277 | 0.0000 | 997 |
| Negated (best patch) | 11.445 | 0.0059 | 17 |

- Negated top predictions: ` one` (0.3711) | ` only` (0.2571) | ` just` (0.1168) | `-` (0.0665) | ` a` (0.0329)
- Patched top predictions: ` the` (0.3145) | ` London` (0.0974) | ` a` (0.0409) | ` Germany` (0.0350) | ` 18` (0.0217)

### pythia-160m

- Outcome: `success`
- SGR: `36.197`
- Best patch: `resid` at layer `3` with Δ logit `+10.188`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 26.328 | 0.0024 | 48 |
| Negated | 16.219 | 0.0000 | 11563 |
| Negated (best patch) | 26.406 | 0.0024 | 59 |

- Negated top predictions: ` only` (0.2440) | ` just` (0.1066) | ` too` (0.0405) | `ori` (0.0380) | ` long` (0.0265)
- Patched top predictions: ` the` (0.1930) | ` a` (0.0282) | ` Germany` (0.0224) | ` London` (0.0182) | ` 18` (0.0123)

### pythia-410m

- Outcome: `success`
- SGR: `7.180`
- Best patch: `resid` at layer `4` with Δ logit `+7.656`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 11.352 | 0.0034 | 26 |
| Negated | 4.078 | 0.0000 | 4020 |
| Negated (best patch) | 11.734 | 0.0053 | 18 |

- Negated top predictions: `oriously` (0.3218) | ` only` (0.2468) | `arial` (0.0712) | `-` (0.0479) | `ary` (0.0435)
- Patched top predictions: ` the` (0.3232) | ` Germany` (0.0632) | ` Berlin` (0.0386) | ` a` (0.0249) | ` Vienna` (0.0207)

## What Helps The Story

- No especially strong shared-case support stood out in this rerun.

## What Hurts The Story

- Case `180` in `pythia-160m` succeeds even with SGR `52.886`.
- Case `180` in `pythia-410m` succeeds even with SGR `4.935`.
- Case `180` does not behave consistently across models, which weakens any one-size-fits-all narrative.
- Case `57` in `gpt2-medium` succeeds even with SGR `2.740`.
- Case `57` in `gpt2-large` succeeds even with SGR `9.264`.
- Case `57` in `pythia-160m` succeeds even with SGR `13.416`.
- Case `57` in `pythia-410m` succeeds even with SGR `2.210`.
- Case `65` in `gpt2-small` succeeds even with SGR `2.010`.
- Case `65` in `gpt2-medium` succeeds even with SGR `7314.642`.
- Case `65` in `gpt2-large` succeeds even with SGR `16.698`.
- Case `65` in `pythia-160m` succeeds even with SGR `36.197`.
- Case `65` in `pythia-410m` succeeds even with SGR `7.180`.
