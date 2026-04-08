# Crossover Analysis Results

This document records the findings from the "Crossover Analysis" experiment proposed in the project mid-submission. The objective of this experiment was to analyse the **crossover layer** — the specific point in the transformer architecture where the `Inhibition (Attn)` signal overtakes the `Retrieval (FFN)` signal, flipping the prediction polarity from the factual entity to a safe, alternate entity.

We wanted to mathematically correlate this crossover point with both the overall Signal-to-Gate Ratio (SGR) and the likelihood of the model hallucinating.

---

## 1. Defining the Crossover Locus
*See `figures/crossover_layer_dist.png`*

By plotting a histogram of every prompt evaluated in `benchmark.py`, we identified the primary locus of logical intervention. 
For GPT-2 Small (12 layers), the logit prediction crossover occurs robustly around **Layer 5 and Layer 6**. 

This suggests that early-mid layers act as the primary "negative logical constraint" injectors. If the constraint isn't successfully applied by this halfway point, the factual FFN representations deeper in the network become increasingly dominant.

## 2. Failure Correlation (The Late Crossover Hazard)
*See `figures/crossover_vs_failure.png`*

We plotted the crossover locus against the binary `negation_failure` flag to test our hypothesis regarding logical timing. 
The data firmly supports the theory: **When the logit-flip / crossover occurs late in the network (e.g., past Layer 8/9), the negation failure probability spikes significantly.**

*   **Success Cases (False):** Crossover uniformly clustered around layers 5-7.
*   **Failure Cases (True):** Crossover stretched into layers 9-11, or failed to occur entirely (resulting in `NaN` and an immediate hallucination).

This confirms the competitive dynamic: "early crossovers" are necessary for safe reasoning. If the semantic retrieval pathway fires first and locks in an attribute, the logic pathway struggles to suppress it at the exit layers.

## 3. SGR Inverse Correlation
*See `figures/crossover_vs_sgr.png`*

Plotting the crossover layer against the core SGR metric (on a log scale) visually proves the underlying mechanics.
*   Samples that crossover extremely early log **SGR < 1** (Logic dominates).
*   Samples that crossover extremely late map to highly positive, runaway SGR values (FFN dominates).

This inverse relationship definitively ties the architectural timing of the attention heads to the magnitude of the hallucination vulnerability measured by the SGR.
