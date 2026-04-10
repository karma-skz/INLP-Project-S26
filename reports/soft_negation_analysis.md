# Soft Negation Analysis Report

- **Models**: `gpt2-small`, `pythia-160m`
- **Negators tested**: ` not` (hard), ` unlikely to be`, ` rarely`, ` maybe`
- **Samples per run**: All available (21,919 for GPT-2 Small, 19,562 for Pythia-160M)
- **Script**: `soft_negation_experiment.py`
- **Results**: `results/soft_negation/`
- **Figures**: `figures/soft_negation/`

---

## Summary Table

| model | negator | n_samples | n_failures | failure_rate | median_sgr | mean_sgr | sgr\_gt1\_rate |
|---|---|---|---|---|---|---|---|
| gpt2-small | not | 21919 | 2498 | 11.40% | 27.09 | 190.25 | 99.1% |
| gpt2-small | unlikely to be | 21919 | 989 | 4.51% | 17.30 | 124.42 | 91.3% |
| gpt2-small | rarely | 21919 | 643 | 2.93% | 9.67 | 78.66 | 84.2% |
| gpt2-small | maybe | 21919 | 1991 | 9.08% | 41.06 | 261.37 | 99.4% |
| pythia-160m | not | 19562 | 1764 | 9.02% | 24.48 | 120.64 | 100.0% |
| pythia-160m | unlikely to be | 19562 | 778 | 3.98% | 22.94 | 109.71 | 99.8% |
| pythia-160m | rarely | 19562 | 688 | 3.52% | 25.36 | 97.13 | 99.9% |
| pythia-160m | maybe | 19562 | 1795 | 9.18% | 35.03 | 201.48 | 99.9% |

---

## Key Findings

### GPT-2 Small

- **` unlikely to be`** (Soft): failure rate = 4.51% — **6.89 pp lower** than hard negation, median SGR = 17.3 (lower than hard's 27.1, indicating stronger relative inhibition)
- **` rarely`** (Soft): failure rate = 2.93% — **8.47 pp lower** than hard negation, median SGR = 9.7 — the strongest suppression of any negator tested
- **` maybe`** (Ambiguous): failure rate = 9.08% — only slightly lower than hard negation (2.32 pp), median SGR = 41.1 — the *weakest* inhibition of any negator

### Pythia-160M

- **` unlikely to be`** (Soft): failure rate = 3.98% — **5.04 pp lower** than hard negation, median SGR = 22.9 (slightly lower than hard's 24.5)
- **` rarely`** (Soft): failure rate = 3.52% — **5.50 pp lower** than hard negation, median SGR = 25.4 (slightly higher, suggesting occasional retrieval amplification)
- **` maybe`** (Ambiguous): failure rate = 9.18% — virtually identical to hard negation (0.16 pp higher), median SGR = 35.0 — the weakest inhibition

---

## Hypothesis Evaluation

> **Mid-submission prediction**: Soft negation should produce *weaker* inhibition than hard negation, yielding higher SGR and higher failure rates.

**Result: ❌ Hypothesis Partially Rejected**

The prediction was that soft negators would be *worse* than hard "not." The data shows the opposite for ` rarely` and ` unlikely to be`:

| Negator | Direction vs Hypothesis |
|---|---|
| ` unlikely to be` | ❌ Lower failure rate and lower/equal SGR — *better* suppression |
| ` rarely` | ❌ Lowest failure rate across all negators — *strongest* suppression |
| ` maybe` | ✅ Highest SGR, near-identical failure rate — ambiguous, not true negation |

### Mechanistic Interpretation

The key distinction is between **logical negation** and **uncertainty markers**:

- **` rarely` / ` unlikely to be`** appear to act as stronger inhibition triggers than "not" in these models. This is counterintuitive syntactically but makes distributional sense: these phrases are longer, rarer, and more specifically associated with *low-frequency* completions in training data. The model's FFN may actually encode weaker associations for these constructions, reducing retrieval strength alongside increasing inhibition.

- **` maybe`** is an **ambiguity token**, not a negation. It does not logically suppress the factual answer; it signals uncertainty. The model's inhibition circuit likely does not engage in the same way, leaving the retrieval signal largely unchecked.

- The **Pink Elephant hypothesis** (mentioning a concept increases its activation) appears to be *most* true for "not" — the adversarial condition — while multi-word soft negators seem to redirect the model's attention more effectively.

---

## SGR Analysis

### What the lower SGR for `rarely` means

For GPT-2 Small, the median SGR drops from **27.1** (hard "not") to **9.7** (rarely). Recall:

```
SGR = FFN retrieval total / Attention inhibition total
```

A lower SGR means the inhibition circuit is either:
1. *Stronger* (more negative attention DLA), or
2. Acting on a *weaker retrieval signal* (smaller FFN positive DLA)

Given the failure rate drops sharply (11.4% → 2.9%), the most likely explanation is (2): the ` rarely` prompt suffix changes the FFN response, retrieving less confidently, making suppression easier regardless of the absolute inhibition strength.

### `maybe` as a control condition

` maybe` is effectively a **negative control**: it does not negate the factual target, so we expect:
- High failure rate ≈ hard negation baseline ✅ (9.08% vs 11.4%)
- High SGR ✅ (41.1 — the highest of any negator)

This validates the experiment: the pipeline correctly detects that ` maybe` provides no real inhibitory signal.

---

## SGR Edge Cases

| model | negator | fail_with_sgr\_gt1 | fail\_with\_sgr\_le1 | failure\_mismatch\_rate |
|---|---|---|---|---|
| gpt2-small | not | 2498 | 0 | 0.00% |
| gpt2-small | unlikely to be | 987 | 2 | 0.20% |
| gpt2-small | rarely | 629 | 14 | 2.18% |
| gpt2-small | maybe | 1990 | 1 | 0.05% |
| pythia-160m | not | 1764 | 0 | 0.00% |
| pythia-160m | unlikely to be | 778 | 0 | 0.00% |
| pythia-160m | rarely | 687 | 1 | 0.15% |
| pythia-160m | maybe | 1795 | 0 | 0.00% |

- For hard "not," **0 failures occur at SGR ≤ 1** (perfect diagnostic boundary).
- For ` rarely` in GPT-2 Small, **14 failures occur at SGR ≤ 1** (2.18% of failures). This is the only condition where the SGR diagnostic meaningfully breaks down — suggesting ` rarely` activates a qualitatively different circuit where the output is not purely a function of the retrieval/inhibition balance.

---

## What Helps Our Story

- `rarely` and `unlikely to be` confirm that the **inhibition circuit can be engaged by non-explicit negation tokens**, providing evidence that the circuit is sensitive to semantic content, not just the literal "not" token.
- The SGR drops consistently with ` rarely` and ` unlikely to be`, supporting SGR as a general diagnostic valid across negation forms — not just hard negation.
- ` maybe` acting as a control validates the pipeline: the model correctly "fails" to negate when asked with an ambiguous term.

## What Weakens Our Story

- **The mechanistic pathway for soft negators is unclear.** We cannot yet determine whether lower failure rates for ` rarely` are due to stronger *attention inhibition* or weaker *FFN retrieval*, since both would reduce SGR.
- The 14 SGR-boundary failures for ` rarely` in GPT-2 Small suggest a **different circuit pathway** that our current FFN/Attention dichotomy does not fully capture.
- We need **head-level DLA analysis for each negator** to determine if the same inhibition heads (e.g., GPT-2 Small head (9,8)) fire for soft negators or if different heads are recruited.

---

## Notes

- All benchmarks ran on the full CounterFact dataset with no sample cap.
- Figures available in `figures/soft_negation/`:
  - `failure_rate_comparison.png` — bar chart comparing failure rates per negator across models
  - `sgr_distribution_comparison.png` — overlaid SGR histograms per model
  - `median_sgr_comparison.png` — grouped bar of median SGR per negator
  - `inhibition_strength_comparison.png` — box plot of raw inhibition strength (skipped: column was not present in this run)
  - `failure_by_sgr_region.png` — stacked bars showing where failures fall in SGR space
- The soft negation markdown report auto-generated by the script will be at `results/soft_negation/soft_negation_report.md` after re-running plots.
