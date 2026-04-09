# Qualitative Analysis

- Results source: `results/gpt2-small_not_benchmark.csv`
- Model rerun: `gpt2-small`
- Cases analysed: 4

## Case 42:  Arun Nehru

- Positive prompt: `The profession of Arun Nehru is`
- Negated prompt: `The profession of Arun Nehru is not`
- Target token: ` politician`
- Benchmark label: `failure`
- SGR: `6.781`

The target becomes more likely after negation (rank 12053 -> 9053), which is a direct negation failure. The SGR of 6.781 suggests retrieval still dominates inhibition.

| Metric | Positive | Negated |
|---|---:|---:|
| Target logit | 2.681 | 3.774 |
| Target prob | 0.0000 | 0.0000 |
| Target rank | 12053 | 9053 |

**Top Predictions: Positive**

| Token | Prob |
|---|---:|
| `' a'` | 0.0780 |
| `' not'` | 0.0484 |
| `' one'` | 0.0425 |
| `' in'` | 0.0267 |
| `' to'` | 0.0255 |
| `' an'` | 0.0240 |
| `' becoming'` | 0.0172 |
| `' being'` | 0.0164 |

**Top Predictions: Negated**

| Token | Prob |
|---|---:|
| `' a'` | 0.0625 |
| `' just'` | 0.0489 |
| `' only'` | 0.0460 |
| `' to'` | 0.0323 |
| `' new'` | 0.0308 |
| `' the'` | 0.0256 |
| `' about'` | 0.0240 |
| `' without'` | 0.0207 |

**Largest Component Shifts (Negated - Positive DLA)**

| Component | Pos DLA | Neg DLA | Delta |
|---|---:|---:|---:|
| `9_mlp_out` | 7.913 | 1.501 | -6.413 |
| `0_mlp_out` | -1.163 | 3.848 | 5.010 |
| `8_mlp_out` | 1.244 | 6.254 | 5.010 |
| `11_mlp_out` | 16.933 | 11.933 | -4.999 |
| `11_attn_out` | 5.913 | 10.131 | 4.218 |
| `4_mlp_out` | -1.307 | 2.456 | 3.763 |
| `3_mlp_out` | -2.489 | 0.981 | 3.470 |
| `5_mlp_out` | -3.168 | -0.150 | 3.018 |

**Per-Layer Summary on the Negated Prompt**

- Strongest retrieval layers (FFN DLA): L11 (11.933), L8 (6.254), L0 (3.848)
- Strongest inhibition layers (Attn DLA): L0 (-2.881), L10 (-0.975), L4 (-0.264)

**Top Inhibition Heads**

| Head | Delta DLA (pos - neg) |
|---|---:|
| `(6, 7)` | 1.893 |
| `(10, 9)` | 1.661 |
| `(8, 7)` | 1.427 |
| `(9, 8)` | 1.025 |
| `(4, 7)` | 0.931 |
| `(7, 5)` | 0.753 |
| `(0, 5)` | 0.455 |
| `(4, 11)` | 0.420 |

- Head heatmap: `figures/qualitative/42-arun-nehru_head_dla_heatmap.png`

## Case 57: Gilles Grimandi

- Positive prompt: `Gilles Grimandi was born in`
- Negated prompt: `Gilles Grimandi was born in not`
- Target token: ` Gap`
- Benchmark label: `success`
- SGR: `0.970`

The negated prompt suppresses the target (rank 3021 -> 35761) and SGR falls below 1, which matches the intended gating story.

| Metric | Positive | Negated |
|---|---:|---:|
| Target logit | 6.159 | -1.539 |
| Target prob | 0.0000 | 0.0000 |
| Target rank | 3021 | 35761 |

**Top Predictions: Positive**

| Token | Prob |
|---|---:|
| `' Paris'` | 0.0529 |
| `' the'` | 0.0439 |
| `' France'` | 0.0161 |
| `' New'` | 0.0131 |
| `' Rome'` | 0.0114 |
| `' Berlin'` | 0.0103 |
| `' Milan'` | 0.0097 |
| `' Geneva'` | 0.0090 |

**Top Predictions: Negated**

| Token | Prob |
|---|---:|
| `' too'` | 0.4992 |
| `' far'` | 0.1850 |
| `' much'` | 0.0722 |
| `' so'` | 0.0619 |
| `'-'` | 0.0375 |
| `' very'` | 0.0180 |
| `' one'` | 0.0162 |
| `' only'` | 0.0146 |

**Largest Component Shifts (Negated - Positive DLA)**

| Component | Pos DLA | Neg DLA | Delta |
|---|---:|---:|---:|
| `11_mlp_out` | 42.403 | 0.752 | -41.652 |
| `10_mlp_out` | 23.451 | -9.498 | -32.949 |
| `9_mlp_out` | 15.356 | -7.222 | -22.578 |
| `7_mlp_out` | 1.884 | -6.972 | -8.856 |
| `4_mlp_out` | 2.274 | -5.303 | -7.578 |
| `3_mlp_out` | 3.821 | -3.564 | -7.385 |
| `5_mlp_out` | 1.655 | -5.621 | -7.276 |
| `11_attn_out` | 8.798 | 3.547 | -5.251 |

**Per-Layer Summary on the Negated Prompt**

- Strongest retrieval layers (FFN DLA): L8 (2.736), L11 (0.752), L1 (-1.340)
- Strongest inhibition layers (Attn DLA): L0 (-2.369), L1 (-0.983), L3 (-0.242)

**Top Inhibition Heads**

| Head | Delta DLA (pos - neg) |
|---|---:|
| `(8, 11)` | 4.367 |
| `(9, 8)` | 3.490 |
| `(10, 0)` | 3.217 |
| `(11, 1)` | 1.863 |
| `(3, 9)` | 1.795 |
| `(3, 3)` | 1.727 |
| `(11, 8)` | 1.151 |
| `(2, 4)` | 1.094 |

- Head heatmap: `figures/qualitative/57-gilles-grimandi_head_dla_heatmap.png`

## Case 87: Gilli Smyth

- Positive prompt: `Gilli Smyth belongs to the organization of`
- Negated prompt: `Gilli Smyth belongs to the organization of not`
- Target token: ` Gong`
- Benchmark label: `success`
- SGR: `1.055`

The negated prompt suppresses the target (rank 16541 -> 30821), but the SGR remains 1.055; this is a useful mismatch case for the metric.

| Metric | Positive | Negated |
|---|---:|---:|
| Target logit | 2.099 | -0.758 |
| Target prob | 0.0000 | 0.0000 |
| Target rank | 16541 | 30821 |

**Top Predictions: Positive**

| Token | Prob |
|---|---:|
| `' the'` | 0.1461 |
| `' a'` | 0.0235 |
| `' women'` | 0.0171 |
| `' "'` | 0.0163 |
| `' people'` | 0.0118 |
| `' American'` | 0.0073 |
| `' those'` | 0.0058 |
| `' professional'` | 0.0055 |

**Top Predictions: Negated**

| Token | Prob |
|---|---:|
| `' only'` | 0.4527 |
| `'-'` | 0.1432 |
| `' just'` | 0.1133 |
| `' one'` | 0.0427 |
| `'ables'` | 0.0255 |
| `'aries'` | 0.0144 |
| `' so'` | 0.0098 |
| `' too'` | 0.0088 |

**Largest Component Shifts (Negated - Positive DLA)**

| Component | Pos DLA | Neg DLA | Delta |
|---|---:|---:|---:|
| `11_mlp_out` | 9.901 | -4.910 | -14.811 |
| `8_mlp_out` | 1.817 | -8.455 | -10.272 |
| `9_mlp_out` | 8.352 | 0.878 | -7.474 |
| `10_attn_out` | 8.969 | 3.188 | -5.781 |
| `11_attn_out` | 9.038 | 3.816 | -5.222 |
| `3_mlp_out` | 2.558 | -2.382 | -4.939 |
| `0_attn_out` | 1.579 | -2.438 | -4.017 |
| `6_mlp_out` | 2.748 | -0.474 | -3.222 |

**Per-Layer Summary on the Negated Prompt**

- Strongest retrieval layers (FFN DLA): L1 (1.697), L10 (1.630), L9 (0.878)
- Strongest inhibition layers (Attn DLA): L0 (-2.438), L1 (-1.033), L4 (-0.481)

**Top Inhibition Heads**

| Head | Delta DLA (pos - neg) |
|---|---:|
| `(10, 10)` | 7.675 |
| `(0, 1)` | 3.509 |
| `(3, 11)` | 1.759 |
| `(11, 1)` | 1.587 |
| `(11, 2)` | 1.484 |
| `(8, 11)` | 1.423 |
| `(9, 9)` | 1.296 |
| `(8, 4)` | 1.059 |

- Head heatmap: `figures/qualitative/87-gilli-smyth_head_dla_heatmap.png`

## Case 180:  Giovanni Battista Riccioli

- Positive prompt: `The domain of activity of Giovanni Battista Riccioli is`
- Negated prompt: `The domain of activity of Giovanni Battista Riccioli is not`
- Target token: ` astronomy`
- Benchmark label: `failure`
- SGR: `8.391`

The target becomes more likely after negation (rank 14438 -> 8976), which is a direct negation failure. The SGR of 8.391 suggests retrieval still dominates inhibition.

| Metric | Positive | Negated |
|---|---:|---:|
| Target logit | 1.761 | 3.171 |
| Target prob | 0.0000 | 0.0000 |
| Target rank | 14438 | 8976 |

**Top Predictions: Positive**

| Token | Prob |
|---|---:|
| `' the'` | 0.0512 |
| `' a'` | 0.0506 |
| `' not'` | 0.0418 |
| `' now'` | 0.0357 |
| `' one'` | 0.0217 |
| `' being'` | 0.0189 |
| `' in'` | 0.0178 |
| `' known'` | 0.0164 |

**Top Predictions: Negated**

| Token | Prob |
|---|---:|
| `' the'` | 0.0631 |
| `' known'` | 0.0564 |
| `' a'` | 0.0449 |
| `' only'` | 0.0413 |
| `' just'` | 0.0240 |
| `' yet'` | 0.0222 |
| `' in'` | 0.0213 |
| `' limited'` | 0.0189 |

**Largest Component Shifts (Negated - Positive DLA)**

| Component | Pos DLA | Neg DLA | Delta |
|---|---:|---:|---:|
| `11_mlp_out` | 5.132 | 13.868 | 8.736 |
| `8_mlp_out` | -1.151 | 2.879 | 4.030 |
| `2_mlp_out` | -2.621 | 1.161 | 3.782 |
| `6_mlp_out` | 3.194 | -0.159 | -3.353 |
| `11_attn_out` | -0.443 | 2.578 | 3.021 |
| `0_attn_out` | -3.438 | -0.936 | 2.503 |
| `10_attn_out` | 2.379 | 4.568 | 2.189 |
| `10_mlp_out` | 5.242 | 7.348 | 2.106 |

**Per-Layer Summary on the Negated Prompt**

- Strongest retrieval layers (FFN DLA): L11 (13.868), L10 (7.348), L9 (4.952)
- Strongest inhibition layers (Attn DLA): L4 (-2.719), L0 (-0.936), L3 (-0.520)

**Top Inhibition Heads**

| Head | Delta DLA (pos - neg) |
|---|---:|
| `(9, 8)` | 1.460 |
| `(11, 11)` | 1.367 |
| `(4, 9)` | 0.828 |
| `(6, 1)` | 0.758 |
| `(3, 6)` | 0.693 |
| `(1, 5)` | 0.688 |
| `(8, 4)` | 0.613 |
| `(0, 3)` | 0.593 |

- Head heatmap: `figures/qualitative/180-giovanni-battista-riccioli_head_dla_heatmap.png`
