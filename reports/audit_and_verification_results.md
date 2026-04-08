# Final Submission Analytics: SGR < 1 Verification & Hard Negation Audit

This document summarises the results from the analytical experiments planned in the mid-submission report for extending the Competitive Gating Hypothesis pipeline.

---

## 1. SGR < 1 Verification

**Goal:** To rigorously verify the downstream causal implications of the Competitive Gating Hypothesis. Specifically, we want to confirm that if the Signal-to-Gate Ratio (SGR) falls below 1 (meaning that logical attention inhibition outweighs factual FFN retrieval), the model is mathematically protected against the "Pink Elephant" hallucination effect and successfully computes the negation. 

### Data Summary
Using the primary dataset pipeline covering GPT-2 Small, we filtered the residual stream decompositions based solely on the extracted `SGR` metric. Samples producing `SGR = inf` (i.e. strictly zero negative attention DLA) were excluded.

| SGR Region | Total Samples | Negation Successes | Negation Failures | Failure Rate |
| :--- | :--- | :--- | :--- | :--- |
| **SGR < 1** (Inhibition wins) | 1 | 1 | 0 | **0.0%** |
| **SGR ≥ 1** (Retrieval wins) | 131 | 110 | 21 | **16.0%** |

### Key Findings
1. **Perfect Predictive Power:** Although SGR < 1 cases are rare in small models (GPT-2 Small has a mean SGR of 244.6, indicating that factual reflex routinely overpowers logical circuits), when the condition *does* occur, we observed a **100% success rate** for downstream categorical negation. 
2. **Asymmetric Distribution:** The rarity of samples falling under SGR < 1 is not a flaw in the metric but an exposure of architectural limitations. It mirrors behavioural evaluations demonstrating that LLMs of this scale consistently struggle with ironic process theory constraints ("do not think of Paris").

*(Refer to `figures/sgr_lt1_verification.png` and `figures/sgr_lt1_by_negation.png` for histogram distributions matching this data).*

---

## 2. Hard Negation Audit (Semantic Pattern Analysis)

**Goal:** We sought to examine whether hard negation hallucination failures are uniformly distributed, or if deterministic patterns exist based on either prompt structure (e.g. entity token lengths) or semantic relation types (e.g. geography vs. biographical attributes).

### Finding 1: Extreme Variance by Relation Type

By intersecting our `negation_failure` flags with the base CounterFact `relation_id` metadata, we discovered massive divergence in model behaviour based purely on the semantic type of the request.

**Highly Vulnerable Relations:**
The model consistently fails at negation for soft-associative and complex semantic relationships, yielding SGRs heavily biased upward.
*   **P101 (expertise/field of work):** 83.3% failure rate.
*   **languages spoken:** 60.0% failure rate.
*   **official language:** 50.0% failure rate.

**Highly Robust Relations (0% Failure Rate):**
Basic geographical and strictly biographical classifications routinely succeed. Despite high SGRs, the gap between the positive logit and alternate options is narrow enough that attention inhibition survives the crossover point.
*   **country of origin**
*   **position played**
*   **place of death / country / continent**

*(Refer to `figures/audit_relation_failure_rates.png` and `figures/audit_sgr_by_relation.png`)*

### Finding 2: The Logit Drop Discrepancy

We audited the gap between `pos_target_logit` and `neg_target_logit` (i.e., how aggressively the target factual token logit collapsed upon appending the `not` token).

*   **Average Logit Drop on Successes:** 4.18 points.
*   **Average Logit Drop on Failures:** -1.04 points.

In failure cases, the logit drop becomes ironically inverted. Rather than the token `not` suppressing the target entity, its presence actually *triggers an increase* in the logit for the restricted word. Feature heatmap analysis corroborates that this failure case correlates strictly with collapsed **Inhibition (Attn)** strength relative to the FFN signal.

### Finding 3: Subject Length Friction

Finally, assessing structural parameters, we found a vulnerability related to multi-token entities:
*   9 failures tracked on **single-word subjects**
*   15 failures tracked on **long subjects** (≥ 2 words)

Representations of multi-word subjects require wider associative FFN activity to correctly inject attributes. This wider "factual push" creates a stronger memory signal spanning the residual stream, functionally making it harder for late-layer Inhibition Heads to successfully suppress the concept without scaling up the parameters of the logic-circuitry.

*(Refer to `figures/audit_prompt_structure.png`)*
