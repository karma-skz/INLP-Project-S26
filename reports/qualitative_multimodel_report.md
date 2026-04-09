# Multi-Model Qualitative Report

- Goal: compare representative negation cases before and after activation patching across all tested models.
- Models: `gpt2-small`, `pythia-160m`
- Cases: 10547, 833, 18559, 4550

## Case 10547: Nur ad-Din al-Bitruji

- Positive prompt: `Nur ad-Din al-Bitruji's domain of activity is`
- Negated prompt: `Nur ad-Din al-Bitruji's domain of activity is not`
- Target token: ` astronomy`

### gpt2-small

- Benchmark label: `failure`
- SGR: `2.752`
- Best patch: `mlp` at layer `1` with Δ logit `+0.550`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | -2.005 | 0.0000 | 36509 |
| Negated (unpatched) | 1.248 | 0.0000 | 17563 |
| Negated (best patch) | 1.798 | 0.0000 | 15378 |

- Negated top predictions: ` only` (0.3732) | ` just` (0.1163) | ` limited` (0.0765) | ` in` (0.0484) | ` restricted` (0.0437)
- Patched top predictions: ` only` (0.3752) | ` just` (0.1065) | ` limited` (0.0821) | ` restricted` (0.0561) | ` too` (0.0251)
- Strongest residual patch layer: `L6` (Δ `-3.158`)
- Strongest MLP patch layer: `L1` (Δ `+0.550`)
- Strongest attention patch layer: `L0` (Δ `+0.514`)

### pythia-160m

- Benchmark label: `success`
- SGR: `39.189`
- Best patch: `resid` at layer `10` with Δ logit `+0.378`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 17.671 | 0.0000 | 22308 |
| Negated (unpatched) | 17.301 | 0.0000 | 22744 |
| Negated (best patch) | 17.680 | 0.0000 | 22227 |

- Negated top predictions: ` a` (0.1010) | ` the` (0.0404) | ` only` (0.0398) | ` limited` (0.0236) | ` yet` (0.0234)
- Patched top predictions: ` the` (0.1078) | ` a` (0.1001) | ` an` (0.0224) | ` to` (0.0223) | ` in` (0.0193)
- Strongest residual patch layer: `L10` (Δ `+0.378`)
- Strongest MLP patch layer: `L9` (Δ `+0.372`)
- Strongest attention patch layer: `L4` (Δ `+0.106`)

## Case 833:  Georg Ernst Stahl

- Positive prompt: `The domain of activity of Georg Ernst Stahl is`
- Negated prompt: `The domain of activity of Georg Ernst Stahl is not`
- Target token: ` chemistry`

### gpt2-small

- Benchmark label: `failure`
- SGR: `8.258`
- Best patch: `attn` at layer `6` with Δ logit `+0.239`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 0.452 | 0.0000 | 22633 |
| Negated (unpatched) | 3.076 | 0.0000 | 9127 |
| Negated (best patch) | 3.315 | 0.0000 | 7739 |

- Negated top predictions: ` the` (0.0620) | ` a` (0.0483) | ` known` (0.0478) | ` only` (0.0315) | ` in` (0.0236)
- Patched top predictions: ` the` (0.0678) | ` a` (0.0491) | ` known` (0.0441) | ` only` (0.0372) | ` in` (0.0247)
- Strongest residual patch layer: `L4` (Δ `-2.509`)
- Strongest MLP patch layer: `L1` (Δ `+0.045`)
- Strongest attention patch layer: `L6` (Δ `+0.239`)

### pythia-160m

- Benchmark label: `success`
- SGR: `42.560`
- Best patch: `mlp` at layer `9` with Δ logit `+0.695`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 19.970 | 0.0000 | 6850 |
| Negated (unpatched) | 19.735 | 0.0000 | 9487 |
| Negated (best patch) | 20.430 | 0.0000 | 7319 |

- Negated top predictions: ` known` (0.1321) | ` yet` (0.0761) | ` a` (0.0560) | ` only` (0.0520) | ` well` (0.0268)
- Patched top predictions: ` known` (0.1349) | ` a` (0.0835) | ` only` (0.0526) | ` yet` (0.0454) | ` the` (0.0429)
- Strongest residual patch layer: `L3` (Δ `+0.285`)
- Strongest MLP patch layer: `L9` (Δ `+0.695`)
- Strongest attention patch layer: `L5` (Δ `+0.402`)

## Case 18559: Lajos Kossuth

- Positive prompt: `Lajos Kossuth found employment in`
- Negated prompt: `Lajos Kossuth found employment in not`
- Target token: ` Budapest`

### gpt2-small

- Benchmark label: `success`
- SGR: `2.646`
- Best patch: `resid` at layer `1` with Δ logit `+11.433`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 8.802 | 0.0006 | 256 |
| Negated (unpatched) | -2.440 | 0.0000 | 41624 |
| Negated (best patch) | 8.993 | 0.0007 | 235 |

- Negated top predictions: ` only` (0.3680) | ` too` (0.1100) | `-` (0.0700) | ` just` (0.0677) | ` one` (0.0328)
- Patched top predictions: ` the` (0.1758) | ` a` (0.0370) | ` New` (0.0152) | ` Germany` (0.0137) | ` London` (0.0092)
- Strongest residual patch layer: `L1` (Δ `+11.433`)
- Strongest MLP patch layer: `L0` (Δ `+9.662`)
- Strongest attention patch layer: `L9` (Δ `+1.460`)

### pythia-160m

- Benchmark label: `success`
- SGR: `9.088`
- Best patch: `resid` at layer `10` with Δ logit `+10.107`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 24.749 | 0.0006 | 181 |
| Negated (unpatched) | 14.658 | 0.0000 | 21100 |
| Negated (best patch) | 24.765 | 0.0006 | 180 |

- Negated top predictions: ` only` (0.2517) | ` just` (0.1120) | ` one` (0.0387) | `ori` (0.0340) | ` too` (0.0331)
- Patched top predictions: ` the` (0.2146) | ` a` (0.0979) | ` an` (0.0185) | ` his` (0.0121) | ` New` (0.0099)
- Strongest residual patch layer: `L10` (Δ `+10.107`)
- Strongest MLP patch layer: `L0` (Δ `+6.878`)
- Strongest attention patch layer: `L3` (Δ `+1.181`)

## Case 4550: Powder Tower

- Positive prompt: `Powder Tower owner`
- Negated prompt: `Powder Tower owner not`
- Target token: ` Prague`

### gpt2-small

- Benchmark label: `success`
- SGR: `0.994`
- Best patch: `resid` at layer `2` with Δ logit `+3.361`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 1.739 | 0.0000 | 14137 |
| Negated (unpatched) | -1.508 | 0.0000 | 35373 |
| Negated (best patch) | 1.853 | 0.0000 | 13616 |

- Negated top predictions: ` happy` (0.0718) | ` sure` (0.0478) | ` only` (0.0441) | ` allowed` (0.0239) | ` afraid` (0.0217)
- Patched top predictions: ` and` (0.0655) | `,` (0.0272) | ` David` (0.0150) | ` Mike` (0.0148) | ` John` (0.0140)
- Strongest residual patch layer: `L2` (Δ `+3.361`)
- Strongest MLP patch layer: `L0` (Δ `+2.936`)
- Strongest attention patch layer: `L2` (Δ `+0.198`)

### pythia-160m

- Benchmark label: `success`
- SGR: `12.171`
- Best patch: `resid` at layer `7` with Δ logit `+1.026`

| Run | Target logit | Target prob | Target rank |
|---|---:|---:|---:|
| Positive | 18.577 | 0.0000 | 10287 |
| Negated (unpatched) | 17.588 | 0.0000 | 14575 |
| Negated (best patch) | 18.615 | 0.0000 | 10127 |

- Negated top predictions: ` to` (0.2385) | ` interested` (0.0440) | ` allowed` (0.0242) | ` in` (0.0193) | ` guilty` (0.0191)
- Patched top predictions: `\n` (0.0525) | ` and` (0.0359) | `,` (0.0289) | ` says` (0.0269) | `:` (0.0183)
- Strongest residual patch layer: `L7` (Δ `+1.026`)
- Strongest MLP patch layer: `L9` (Δ `+0.348`)
- Strongest attention patch layer: `L8` (Δ `+0.383`)
