# Detailed qualitative analysis from cross-model outputs

This report uses the benchmark outputs in results/cross_model and compares the same 10 case IDs across both models.

## Selected shared case IDs

8676, 13838, 14748, 3500, 18316, 17579, 1485, 18559, 2295, 12009

## Global artifacts

- [../results/qualitative/qualitative_selected_case_ids.csv](../results/qualitative/qualitative_selected_case_ids.csv)
- [../results/qualitative/qualitative_selected_case_metrics.csv](../results/qualitative/qualitative_selected_case_metrics.csv)
- [../figures/qualitative/qualitative_case_comparison.png](../figures/qualitative/qualitative_case_comparison.png)
- [../figures/qualitative/qualitative_layer_shift_heatmap.png](../figures/qualitative/qualitative_layer_shift_heatmap.png)

## Case-by-case analysis

### Case 8676

- Subject: Lydia Field Emmet
- Positive prompt: Lydia Field Emmet, who plays
- Negated prompt: Lydia Field Emmet, who plays not
- Target token: ' portrait'

Artifacts:

- [../results/qualitative/qualitative_cases/case_8676/case_metrics.csv](../results/qualitative/qualitative_cases/case_8676/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_8676/layer_metrics.csv](../results/qualitative/qualitative_cases/case_8676/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_8676/layer_concentration.csv](../results/qualitative/qualitative_cases/case_8676/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_8676/layer_behavior.png](../figures/qualitative/qualitative_cases/case_8676/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_8676/behavior_summary.png](../figures/qualitative/qualitative_cases/case_8676/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=False, SGR=0.012, rank drop=26148, best patch=attn@L9, patch rank recovery=0.
- pythia-160m: failure=True, SGR=271.114, rank drop=-7432, best patch=mlp@L10, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 11 (Δ=-25.582), top-3 layers explain 60.5% of absolute shift.
- pythia-160m: dominant shift at layer 11 (Δ=-34.348), top-3 layers explain 80.2% of absolute shift.

### Case 13838

- Subject: bpost
- Positive prompt: bpost, by
- Negated prompt: bpost, by not
- Target token: ' Belgium'

Artifacts:

- [../results/qualitative/qualitative_cases/case_13838/case_metrics.csv](../results/qualitative/qualitative_cases/case_13838/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_13838/layer_metrics.csv](../results/qualitative/qualitative_cases/case_13838/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_13838/layer_concentration.csv](../results/qualitative/qualitative_cases/case_13838/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_13838/layer_behavior.png](../figures/qualitative/qualitative_cases/case_13838/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_13838/behavior_summary.png](../figures/qualitative/qualitative_cases/case_13838/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=False, SGR=0.891, rank drop=23050, best patch=mlp@L8, patch rank recovery=0.
- pythia-160m: failure=True, SGR=20.025, rank drop=-9618, best patch=mlp@L11, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 0 (Δ=-8.223), top-3 layers explain 45.4% of absolute shift.
- pythia-160m: dominant shift at layer 11 (Δ=22.145), top-3 layers explain 76.1% of absolute shift.

### Case 14748

- Subject: James Squire
- Positive prompt: James Squire, from
- Negated prompt: James Squire, from not
- Target token: ' Lion'

Artifacts:

- [../results/qualitative/qualitative_cases/case_14748/case_metrics.csv](../results/qualitative/qualitative_cases/case_14748/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_14748/layer_metrics.csv](../results/qualitative/qualitative_cases/case_14748/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_14748/layer_concentration.csv](../results/qualitative/qualitative_cases/case_14748/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_14748/layer_behavior.png](../figures/qualitative/qualitative_cases/case_14748/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_14748/behavior_summary.png](../figures/qualitative/qualitative_cases/case_14748/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=False, SGR=0.327, rank drop=38946, best patch=attn@L10, patch rank recovery=0.
- pythia-160m: failure=False, SGR=8.191, rank drop=6436, best patch=mlp@L4, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 11 (Δ=-22.771), top-3 layers explain 43.5% of absolute shift.
- pythia-160m: dominant shift at layer 11 (Δ=-27.980), top-3 layers explain 77.2% of absolute shift.

### Case 3500

- Subject:  Yalkut Yosef
- Positive prompt: The original language of Yalkut Yosef is
- Negated prompt: The original language of Yalkut Yosef is not
- Target token: ' Hebrew'

Artifacts:

- [../results/qualitative/qualitative_cases/case_3500/case_metrics.csv](../results/qualitative/qualitative_cases/case_3500/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_3500/layer_metrics.csv](../results/qualitative/qualitative_cases/case_3500/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_3500/layer_concentration.csv](../results/qualitative/qualitative_cases/case_3500/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_3500/layer_behavior.png](../figures/qualitative/qualitative_cases/case_3500/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_3500/behavior_summary.png](../figures/qualitative/qualitative_cases/case_3500/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=True, SGR=nan, rank drop=-2, best patch=mlp@L11, patch rank recovery=0.
- pythia-160m: failure=True, SGR=12.608, rank drop=-1, best patch=mlp@L9, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 11 (Δ=23.898), top-3 layers explain 74.6% of absolute shift.
- pythia-160m: dominant shift at layer 11 (Δ=-5.559), top-3 layers explain 55.0% of absolute shift.

### Case 18316

- Subject:  Urwah ibn Mas'ud
- Positive prompt: The official religion of Urwah ibn Mas'ud is
- Negated prompt: The official religion of Urwah ibn Mas'ud is not
- Target token: ' Islam'

Artifacts:

- [../results/qualitative/qualitative_cases/case_18316/case_metrics.csv](../results/qualitative/qualitative_cases/case_18316/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_18316/layer_metrics.csv](../results/qualitative/qualitative_cases/case_18316/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_18316/layer_concentration.csv](../results/qualitative/qualitative_cases/case_18316/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_18316/layer_behavior.png](../figures/qualitative/qualitative_cases/case_18316/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_18316/behavior_summary.png](../figures/qualitative/qualitative_cases/case_18316/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=True, SGR=65.873, rank drop=-1, best patch=attn@L11, patch rank recovery=0.
- pythia-160m: failure=True, SGR=85.051, rank drop=-2, best patch=mlp@L10, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 8 (Δ=-30.703), top-3 layers explain 78.2% of absolute shift.
- pythia-160m: dominant shift at layer 9 (Δ=-5.816), top-3 layers explain 62.5% of absolute shift.

### Case 17579

- Subject:  German-speaking Community of Belgium
- Positive prompt: In German-speaking Community of Belgium, the language spoken is
- Negated prompt: In German-speaking Community of Belgium, the language spoken is not
- Target token: ' German'

Artifacts:

- [../results/qualitative/qualitative_cases/case_17579/case_metrics.csv](../results/qualitative/qualitative_cases/case_17579/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_17579/layer_metrics.csv](../results/qualitative/qualitative_cases/case_17579/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_17579/layer_concentration.csv](../results/qualitative/qualitative_cases/case_17579/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_17579/layer_behavior.png](../figures/qualitative/qualitative_cases/case_17579/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_17579/behavior_summary.png](../figures/qualitative/qualitative_cases/case_17579/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=True, SGR=2896.223, rank drop=-2, best patch=mlp@L11, patch rank recovery=0.
- pythia-160m: failure=True, SGR=22.082, rank drop=-1, best patch=mlp@L9, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 11 (Δ=34.613), top-3 layers explain 78.0% of absolute shift.
- pythia-160m: dominant shift at layer 8 (Δ=-5.256), top-3 layers explain 58.3% of absolute shift.

### Case 1485

- Subject: Cyril Lemoine
- Positive prompt: Cyril Lemoine is native to
- Negated prompt: Cyril Lemoine is native to not
- Target token: ' Tours'

Artifacts:

- [../results/qualitative/qualitative_cases/case_1485/case_metrics.csv](../results/qualitative/qualitative_cases/case_1485/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_1485/layer_metrics.csv](../results/qualitative/qualitative_cases/case_1485/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_1485/layer_concentration.csv](../results/qualitative/qualitative_cases/case_1485/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_1485/layer_behavior.png](../figures/qualitative/qualitative_cases/case_1485/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_1485/behavior_summary.png](../figures/qualitative/qualitative_cases/case_1485/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=False, SGR=1.206, rank drop=38555, best patch=attn@L6, patch rank recovery=0.
- pythia-160m: failure=False, SGR=2.815, rank drop=30841, best patch=attn@L3, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 10 (Δ=-22.650), top-3 layers explain 49.9% of absolute shift.
- pythia-160m: dominant shift at layer 11 (Δ=-40.762), top-3 layers explain 78.5% of absolute shift.

### Case 18559

- Subject: Lajos Kossuth
- Positive prompt: Lajos Kossuth found employment in
- Negated prompt: Lajos Kossuth found employment in not
- Target token: ' Budapest'

Artifacts:

- [../results/qualitative/qualitative_cases/case_18559/case_metrics.csv](../results/qualitative/qualitative_cases/case_18559/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_18559/layer_metrics.csv](../results/qualitative/qualitative_cases/case_18559/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_18559/layer_concentration.csv](../results/qualitative/qualitative_cases/case_18559/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_18559/layer_behavior.png](../figures/qualitative/qualitative_cases/case_18559/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_18559/behavior_summary.png](../figures/qualitative/qualitative_cases/case_18559/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=False, SGR=2.647, rank drop=41436, best patch=attn@L6, patch rank recovery=0.
- pythia-160m: failure=False, SGR=9.069, rank drop=20720, best patch=mlp@L5, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 9 (Δ=-44.926), top-3 layers explain 61.3% of absolute shift.
- pythia-160m: dominant shift at layer 11 (Δ=-37.551), top-3 layers explain 79.6% of absolute shift.

### Case 2295

- Subject: John Feaver
- Positive prompt: John Feaver is originally from
- Negated prompt: John Feaver is originally from not
- Target token: ' Fleet'

Artifacts:

- [../results/qualitative/qualitative_cases/case_2295/case_metrics.csv](../results/qualitative/qualitative_cases/case_2295/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_2295/layer_metrics.csv](../results/qualitative/qualitative_cases/case_2295/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_2295/layer_concentration.csv](../results/qualitative/qualitative_cases/case_2295/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_2295/layer_behavior.png](../figures/qualitative/qualitative_cases/case_2295/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_2295/behavior_summary.png](../figures/qualitative/qualitative_cases/case_2295/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=False, SGR=0.000, rank drop=40004, best patch=attn@L4, patch rank recovery=0.
- pythia-160m: failure=False, SGR=5.139, rank drop=20926, best patch=mlp@L3, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 10 (Δ=-46.049), top-3 layers explain 61.0% of absolute shift.
- pythia-160m: dominant shift at layer 11 (Δ=-47.891), top-3 layers explain 81.0% of absolute shift.

### Case 12009

- Subject: Max Brod
- Positive prompt: Max Brod was employed in
- Negated prompt: Max Brod was employed in not
- Target token: ' Prague'

Artifacts:

- [../results/qualitative/qualitative_cases/case_12009/case_metrics.csv](../results/qualitative/qualitative_cases/case_12009/case_metrics.csv)
- [../results/qualitative/qualitative_cases/case_12009/layer_metrics.csv](../results/qualitative/qualitative_cases/case_12009/layer_metrics.csv)
- [../results/qualitative/qualitative_cases/case_12009/layer_concentration.csv](../results/qualitative/qualitative_cases/case_12009/layer_concentration.csv)
- [../figures/qualitative/qualitative_cases/case_12009/layer_behavior.png](../figures/qualitative/qualitative_cases/case_12009/layer_behavior.png)
- [../figures/qualitative/qualitative_cases/case_12009/behavior_summary.png](../figures/qualitative/qualitative_cases/case_12009/behavior_summary.png)

Model-wise behavioral notes:

- gpt2-small: failure=False, SGR=0.661, rank drop=34533, best patch=attn@L10, patch rank recovery=0.
- pythia-160m: failure=False, SGR=13.811, rank drop=25543, best patch=mlp@L5, patch rank recovery=0.

Layer-shift concentration notes:

- gpt2-small: dominant shift at layer 10 (Δ=-31.464), top-3 layers explain 58.2% of absolute shift.
- pythia-160m: dominant shift at layer 11 (Δ=-42.484), top-3 layers explain 72.7% of absolute shift.
