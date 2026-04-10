# Qualitative Analysis

- Models: `gpt2-small`, `gpt2-medium`, `gpt2-large`, `pythia-160m`, `pythia-410m`
- Results source: `results/cross_model`
- Negator suffix: ` not`
- Cases per model: `3`

## gpt2-small

### Case 180:  Giovanni Battista Riccioli

- Bucket: `largest_failure`
- Positive prompt: `The domain of activity of Giovanni Battista Riccioli is`
- Negated prompt: `The domain of activity of Giovanni Battista Riccioli is not`
- Target token: ` astronomy`
- Outcome: `failure`
- SGR: `8.475`
- Best patch: `attn` at layer `6` with Δ logit `+0.203`
- Reading: The target climbs after negation (rank 14370 -> 8908), so this is a direct failure. The strongest patch is attn at layer 6 with Δ logit +0.203.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 1.774 | 0.0000 | 14370 |
| Negated | 3.191 | 0.0000 | 8908 |
| Negated (best patch) | 3.395 | 0.0000 | 7661 |

- Positive top predictions: ` the` (0.0517) | ` a` (0.0509) | ` not` (0.0425) | ` now` (0.0361) | ` one` (0.0219)
- Negated top predictions: ` the` (0.0638) | ` known` (0.0572) | ` a` (0.0456) | ` only` (0.0421) | ` just` (0.0244)
- Patched top predictions: ` the` (0.0704) | ` only` (0.0531) | ` known` (0.0507) | ` a` (0.0473) | ` just` (0.0310)

### Case 57: Gilles Grimandi

- Bucket: `clean_success`
- Positive prompt: `Gilles Grimandi was born in`
- Negated prompt: `Gilles Grimandi was born in not`
- Target token: ` Gap`
- Outcome: `success`
- SGR: `0.967`
- Best patch: `resid` at layer `10` with Δ logit `+7.695`
- Reading: The target is suppressed cleanly (rank 3021 -> 35724) and the case agrees with the main story. The best patch still appears at resid layer 10.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 6.164 | 0.0000 | 3021 |
| Negated | -1.527 | 0.0000 | 35724 |
| Negated (best patch) | 6.168 | 0.0000 | 3030 |

- Positive top predictions: ` Paris` (0.0539) | ` the` (0.0447) | ` France` (0.0159) | ` New` (0.0132) | ` Rome` (0.0116)
- Negated top predictions: ` too` (0.5024) | ` far` (0.1848) | ` much` (0.0712) | ` so` (0.0619) | `-` (0.0367)
- Patched top predictions: ` Paris` (0.0564) | ` the` (0.0446) | ` France` (0.0157) | ` New` (0.0132) | ` Rome` (0.0116)

### Case 87: Gilli Smyth

- Bucket: `sgr_mismatch`
- Positive prompt: `Gilli Smyth belongs to the organization of`
- Negated prompt: `Gilli Smyth belongs to the organization of not`
- Target token: ` Gong`
- Outcome: `success`
- SGR: `1.069`
- Best patch: `resid` at layer `11` with Δ logit `+2.858`
- Reading: The model suppresses the target (rank 16502 -> 30839), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 2.104 | 0.0000 | 16502 |
| Negated | -0.754 | 0.0000 | 30839 |
| Negated (best patch) | 2.104 | 0.0000 | 16502 |

- Positive top predictions: ` the` (0.1473) | ` a` (0.0237) | ` women` (0.0170) | ` "` (0.0165) | ` people` (0.0117)
- Negated top predictions: ` only` (0.4551) | `-` (0.1421) | ` just` (0.1133) | ` one` (0.0427) | `ables` (0.0257)
- Patched top predictions: ` the` (0.1473) | ` a` (0.0237) | ` women` (0.0170) | ` "` (0.0165) | ` people` (0.0117)

### What Helps The Story

- Case `180` is a true failure and still admits a positive patch effect (+0.203).

### What Hurts The Story

- Case `87` succeeds despite SGR `1.069`, showing the metric is not decisive on its own.

## gpt2-medium

### Case 136:  Domingo de Soto

- Bucket: `largest_failure`
- Positive prompt: `The expertise of Domingo de Soto is`
- Negated prompt: `The expertise of Domingo de Soto is not`
- Target token: ` theology`
- Outcome: `failure`
- SGR: `46.335`
- Best patch: `mlp` at layer `20` with Δ logit `+0.477`
- Reading: The target climbs after negation (rank 16004 -> 5525), so this is a direct failure. The strongest patch is mlp at layer 20 with Δ logit +0.477.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 1.712 | 0.0000 | 16004 |
| Negated | 4.508 | 0.0000 | 5525 |
| Negated (best patch) | 4.984 | 0.0000 | 4536 |

- Positive top predictions: ` recognized` (0.0504) | ` a` (0.0328) | ` the` (0.0258) | ` unmatched` (0.0210) | ` in` (0.0201)
- Negated top predictions: ` limited` (0.2463) | ` only` (0.1979) | ` just` (0.1238) | ` confined` (0.0838) | ` restricted` (0.0283)
- Patched top predictions: ` only` (0.2156) | ` limited` (0.1678) | ` just` (0.1414) | ` confined` (0.0470) | ` in` (0.0374)

### Case 57: Gilles Grimandi

- Bucket: `clean_success`
- Positive prompt: `Gilles Grimandi was born in`
- Negated prompt: `Gilles Grimandi was born in not`
- Target token: ` Gap`
- Outcome: `success`
- SGR: `2.740`
- Best patch: `resid` at layer `2` with Δ logit `+6.946`
- Reading: The model suppresses the target (rank 2984 -> 33624), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 5.691 | 0.0000 | 2984 |
| Negated | -1.126 | 0.0000 | 33624 |
| Negated (best patch) | 5.820 | 0.0000 | 2860 |

- Positive top predictions: ` Paris` (0.0832) | ` the` (0.0390) | ` France` (0.0306) | ` Switzerland` (0.0126) | ` T` (0.0123)
- Negated top predictions: ` far` (0.3210) | ` too` (0.1227) | `-` (0.0775) | ` one` (0.0617) | ` a` (0.0589)
- Patched top predictions: ` Paris` (0.0750) | ` the` (0.0431) | ` France` (0.0255) | ` B` (0.0142) | ` T` (0.0135)

### Case 31: controller.controller

- Bucket: `sgr_mismatch`
- Positive prompt: `controller.controller, that originated in`
- Negated prompt: `controller.controller, that originated in not`
- Target token: ` Canada`
- Outcome: `success`
- SGR: `2.321`
- Best patch: `resid` at layer `2` with Δ logit `+4.318`
- Reading: The model suppresses the target (rank 1634 -> 14601), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 5.176 | 0.0001 | 1634 |
| Negated | 1.053 | 0.0000 | 14601 |
| Negated (best patch) | 5.371 | 0.0001 | 1455 |

- Positive top predictions: ` the` (0.1963) | ` a` (0.0285) | ` Angular` (0.0222) | ` this` (0.0196) | ` Rails` (0.0121)
- Negated top predictions: `epad` (0.1334) | `-` (0.1234) | `h` (0.0461) | ` only` (0.0291) | ` a` (0.0178)
- Patched top predictions: ` the` (0.1873) | ` a` (0.0266) | ` Angular` (0.0253) | ` this` (0.0147) | ` Rails` (0.0128)

### What Helps The Story

- Case `136` is a true failure and still admits a positive patch effect (+0.477).

### What Hurts The Story

- Case `57` succeeds despite SGR `2.740`, showing the metric is not decisive on its own.
- Case `31` succeeds despite SGR `2.321`, showing the metric is not decisive on its own.

## gpt2-large

### Case 90:  Rabat

- Bucket: `largest_failure`
- Positive prompt: `The twin city of Rabat is`
- Negated prompt: `The twin city of Rabat is not`
- Target token: ` Damascus`
- Outcome: `failure`
- SGR: `61.805`
- Best patch: `mlp` at layer `18` with Δ logit `+0.270`
- Reading: The target climbs after negation (rank 9116 -> 2225), so this is a direct failure. The strongest patch is mlp at layer 18 with Δ logit +0.270.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 2.922 | 0.0000 | 9116 |
| Negated | 5.531 | 0.0000 | 2225 |
| Negated (best patch) | 5.801 | 0.0000 | 1926 |

- Positive top predictions: ` a` (0.1416) | ` home` (0.0950) | ` the` (0.0921) | ` one` (0.0628) | ` known` (0.0550)
- Negated top predictions: ` only` (0.1486) | ` a` (0.1292) | ` known` (0.0937) | ` the` (0.0937) | ` far` (0.0577)
- Patched top predictions: ` only` (0.1630) | ` a` (0.1310) | ` the` (0.0929) | ` known` (0.0907) | ` just` (0.0563)

### Case 57: Gilles Grimandi

- Bucket: `clean_success`
- Positive prompt: `Gilles Grimandi was born in`
- Negated prompt: `Gilles Grimandi was born in not`
- Target token: ` Gap`
- Outcome: `success`
- SGR: `9.264`
- Best patch: `resid` at layer `1` with Δ logit `+6.034`
- Reading: The model suppresses the target (rank 3187 -> 30482), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 4.953 | 0.0000 | 3187 |
| Negated | -0.792 | 0.0000 | 30482 |
| Negated (best patch) | 5.242 | 0.0000 | 2798 |

- Positive top predictions: ` Paris` (0.1556) | ` the` (0.0497) | ` France` (0.0277) | ` Montreal` (0.0165) | ` 18` (0.0121)
- Negated top predictions: ` far` (0.2346) | `hern` (0.1368) | `-` (0.1327) | ` one` (0.0417) | ` too` (0.0330)
- Patched top predictions: ` Paris` (0.1609) | ` the` (0.0518) | ` France` (0.0291) | ` Montreal` (0.0144) | ` B` (0.0123)

### Case 169:  Patrick Manson

- Bucket: `sgr_mismatch`
- Positive prompt: `The occupation of Patrick Manson is`
- Negated prompt: `The occupation of Patrick Manson is not`
- Target token: ` physician`
- Outcome: `success`
- SGR: `2.471`
- Best patch: `resid` at layer `22` with Δ logit `+1.444`
- Reading: The model suppresses the target (rank 13437 -> 22489), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 1.791 | 0.0000 | 13437 |
| Negated | 0.365 | 0.0000 | 22489 |
| Negated (best patch) | 1.810 | 0.0000 | 13406 |

- Positive top predictions: ` a` (0.1537) | ` the` (0.0698) | ` one` (0.0651) | ` an` (0.0458) | ` not` (0.0430)
- Negated top predictions: ` a` (0.1310) | ` the` (0.0879) | ` over` (0.0801) | ` just` (0.0701) | ` only` (0.0586)
- Patched top predictions: ` a` (0.1532) | ` the` (0.0690) | ` one` (0.0664) | ` an` (0.0453) | ` not` (0.0432)

### What Helps The Story

- Case `90` is a true failure and still admits a positive patch effect (+0.270).

### What Hurts The Story

- Case `57` succeeds despite SGR `9.264`, showing the metric is not decisive on its own.
- Case `169` succeeds despite SGR `2.471`, showing the metric is not decisive on its own.

## pythia-160m

### Case 42:  Arun Nehru

- Bucket: `largest_failure`
- Positive prompt: `The profession of Arun Nehru is`
- Negated prompt: `The profession of Arun Nehru is not`
- Target token: ` politician`
- Outcome: `failure`
- SGR: `30.689`
- Best patch: `mlp` at layer `4` with Δ logit `+0.531`
- Reading: The target climbs after negation (rank 17204 -> 13297), so this is a direct failure. The strongest patch is mlp at layer 4 with Δ logit +0.531.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 18.500 | 0.0000 | 17204 |
| Negated | 19.156 | 0.0000 | 13297 |
| Negated (best patch) | 19.688 | 0.0000 | 9422 |

- Positive top predictions: ` a` (0.1414) | ` the` (0.0647) | ` one` (0.0489) | ` an` (0.0369) | ` not` (0.0242)
- Negated top predictions: ` only` (0.1420) | ` a` (0.0889) | ` just` (0.0547) | ` limited` (0.0483) | ` the` (0.0317)
- Patched top predictions: ` a` (0.1375) | ` only` (0.1213) | ` just` (0.0714) | ` the` (0.0514) | ` to` (0.0353)

### Case 211: Tankred Dorst

- Bucket: `clean_success`
- Positive prompt: `Tankred Dorst found employment in`
- Negated prompt: `Tankred Dorst found employment in not`
- Target token: ` Munich`
- Outcome: `success`
- SGR: `13.980`
- Best patch: `resid` at layer `8` with Δ logit `+8.672`
- Reading: The model suppresses the target (rank 306 -> 20754), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 23.797 | 0.0003 | 306 |
| Negated | 15.125 | 0.0000 | 20754 |
| Negated (best patch) | 23.797 | 0.0003 | 309 |

- Positive top predictions: ` the` (0.2218) | ` a` (0.0527) | ` an` (0.0109) | ` 2006` (0.0082) | ` 2009` (0.0079)
- Negated top predictions: ` only` (0.2686) | ` just` (0.0872) | ` one` (0.0358) | ` too` (0.0311) | ` long` (0.0262)
- Patched top predictions: ` the` (0.2251) | ` a` (0.0518) | ` an` (0.0107) | ` 2006` (0.0081) | ` 2009` (0.0079)

### Case 160: Chevrolet Greenbrier

- Bucket: `sgr_mismatch`
- Positive prompt: `Chevrolet Greenbrier, created by`
- Negated prompt: `Chevrolet Greenbrier, created by not`
- Target token: ` Chevrolet`
- Outcome: `success`
- SGR: `5.331`
- Best patch: `resid` at layer `7` with Δ logit `+8.609`
- Reading: The model suppresses the target (rank 6 -> 2368), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 27.750 | 0.0121 | 6 |
| Negated | 19.234 | 0.0000 | 2368 |
| Negated (best patch) | 27.844 | 0.0136 | 6 |

- Positive top predictions: ` the` (0.0720) | ` Che` (0.0220) | ` a` (0.0179) | ` former` (0.0168) | ` Jeff` (0.0133)
- Negated top predictions: ` only` (0.2693) | `-` (0.1487) | ` being` (0.0498) | ` just` (0.0232) | ` having` (0.0192)
- Patched top predictions: ` the` (0.0713) | ` Che` (0.0235) | ` a` (0.0177) | ` former` (0.0169) | ` Jeff` (0.0138)

### What Helps The Story

- Case `42` is a true failure and still admits a positive patch effect (+0.531).

### What Hurts The Story

- Case `211` succeeds despite SGR `13.980`, showing the metric is not decisive on its own.
- Case `160` succeeds despite SGR `5.331`, showing the metric is not decisive on its own.

## pythia-410m

### Case 136:  Domingo de Soto

- Bucket: `largest_failure`
- Positive prompt: `The expertise of Domingo de Soto is`
- Negated prompt: `The expertise of Domingo de Soto is not`
- Target token: ` theology`
- Outcome: `failure`
- SGR: `2.701`
- Best patch: `mlp` at layer `7` with Δ logit `+0.848`
- Reading: The target climbs after negation (rank 15575 -> 8028), so this is a direct failure. The strongest patch is mlp at layer 7 with Δ logit +0.848.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 1.272 | 0.0000 | 15575 |
| Negated | 2.980 | 0.0000 | 8028 |
| Negated (best patch) | 3.828 | 0.0000 | 5674 |

- Positive top predictions: ` un` (0.0532) | ` a` (0.0489) | ` the` (0.0372) | ` not` (0.0238) | ` in` (0.0229)
- Negated top predictions: ` limited` (0.2329) | ` only` (0.1266) | ` the` (0.0431) | ` in` (0.0371) | ` a` (0.0325)
- Patched top predictions: ` limited` (0.2983) | ` only` (0.1177) | ` the` (0.0440) | ` just` (0.0397) | ` a` (0.0359)

### Case 15: Percy Snow

- Bucket: `clean_success`
- Positive prompt: `Percy Snow, the`
- Negated prompt: `Percy Snow, the not`
- Target token: ` linebacker`
- Outcome: `success`
- SGR: `7.644`
- Best patch: `resid` at layer `14` with Δ logit `+6.051`
- Reading: The model suppresses the target (rank 5281 -> 34034), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 4.531 | 0.0000 | 5281 |
| Negated | -1.422 | 0.0000 | 34034 |
| Negated (best patch) | 4.629 | 0.0000 | 5080 |

- Positive top predictions: ` man` (0.0529) | ` son` (0.0215) | ` former` (0.0180) | ` first` (0.0149) | ` founder` (0.0126)
- Negated top predictions: `oriously` (0.5020) | `-` (0.3506) | ` so` (0.0378) | `ori` (0.0096) | ` quite` (0.0086)
- Patched top predictions: ` man` (0.0613) | ` son` (0.0222) | ` former` (0.0182) | ` first` (0.0145) | ` founder` (0.0129)

### Case 133: Joseph Wostinholm

- Bucket: `sgr_mismatch`
- Positive prompt: `Joseph Wostinholm died at`
- Negated prompt: `Joseph Wostinholm died at not`
- Target token: ` Sheffield`
- Outcome: `success`
- SGR: `1.428`
- Best patch: `resid` at layer `15` with Δ logit `+7.325`
- Reading: The model suppresses the target (rank 1432 -> 27643), but SGR stays above 1. This is a mismatch case and weakens any strict threshold interpretation.

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 6.781 | 0.0000 | 1432 |
| Negated | -0.462 | 0.0000 | 27643 |
| Negated (best patch) | 6.863 | 0.0000 | 1415 |

- Positive top predictions: ` the` (0.2223) | ` age` (0.1931) | ` his` (0.1144) | ` home` (0.0668) | ` a` (0.0346)
- Negated top predictions: ` far` (0.2664) | ` more` (0.1760) | ` too` (0.0851) | ` long` (0.0638) | ` much` (0.0550)
- Patched top predictions: ` age` (0.1902) | ` the` (0.1787) | ` his` (0.1093) | ` home` (0.0762) | ` a` (0.0352)

### What Helps The Story

- Case `136` is a true failure and still admits a positive patch effect (+0.848).

### What Hurts The Story

- Case `15` succeeds despite SGR `7.644`, showing the metric is not decisive on its own.
- Case `133` succeeds despite SGR `1.428`, showing the metric is not decisive on its own.
