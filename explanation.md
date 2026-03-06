## How the Pipeline Works (Phase by Phase)

### Phase 0 — Setup

Loads GPT-2 Small via TransformerLens and defines a single prompt pair:

- **Positive**: `"The capital of France is"` → should predict `" Paris"`
- **Negated**: `"The capital of France is not"` → should NOT predict `" Paris"`

### Phase 1 — Behavioural Comparison

Runs both prompts and checks the surface-level output: What does the model predict? Where does `" Paris"` rank? If `" Paris"` is still in the top few tokens for the negated prompt, **that's a negation failure** — the "Pink Elephant" effect in action.

### Phase 2 — Residual Stream Decomposition

The residual stream is the transformer's information highway. Every component (embedding, attention layer, FFN layer) **adds** its output to this stream. `cache.decompose_resid()` breaks the final residual stream into each component's individual contribution. This tells us "who wrote what" into the model's working memory.

### Phase 3 — Direct Logit Attribution (DLA) ⭐ Core Technique

This is the heart of your project. The final logit for any token is:

$$\text{logit}(t) = \mathbf{r}_{\text{final}} \cdot \mathbf{W}_U[:, t]$$

Since the residual stream is a **sum** of all component outputs, we can attribute the final logit to each component:

$$\text{logit}(t) = \sum_{c} \mathbf{r}_c \cdot \mathbf{W}_U[:, t]$$

- **Positive DLA** = that component pushed the model **toward** predicting `" Paris"`
- **Negative DLA** = that component pushed the model **away from** `" Paris"`

### Phase 4 — Memory vs Inhibition Separation

We split components into two camps:

- **FFN layers** = Memory/Retrieval (Geva et al.'s "key-value stores")
- **Attention layers** = Logic/Inhibition (Hanna et al.'s "inhibition heads")

**Your key hypothesis**: The FFN DLA values should be **nearly identical** between positive and negated prompts (retrieval is "logic-blind"), while the attention DLA should **differ dramatically** (this is where negation processing happens, or fails to).

### Phase 5 — Signal-to-Gate Ratio (SGR) ⭐ Your Novel Metric

$$\text{SGR} = \frac{|\text{FFN layers pushing toward target}|}{|\text{Attn layers pushing away from target}|}$$

- $\text{SGR} > 1$ → Memory overwhelms logic → **hallucination/negation failure**
- $\text{SGR} < 1$ → Logic successfully suppresses → **correct behaviour**

### Phase 5b — Crossover Point

Plots the cumulative DLA layer by layer. The **crossover point** is the layer where cumulative inhibition finally overcomes cumulative retrieval — or fails to, confirming the negation failure.

### Phase 6 — Activation Patching (Causal Intervention)

For each layer, we **swap** the negated-prompt activation with the positive-prompt activation and measure the change in target logit. This answers: _"Which specific layers are causally responsible for negation processing?"_

- Large **positive Δ** from patching = that layer was doing important negation work (removing its negation logic lets the target logit rise)
- We do this separately for **residual stream**, **MLP outputs**, and **attention outputs**

### Phase 7 — Visualization

Saves 5 publication-quality figures to `figures/`:

1. FFN DLA comparison (positive vs negated)
2. Attention DLA comparison
3. Cumulative DLA with crossover point
4. Activation patching results (3-panel)
5. SGR bar chart

---

## How to Proceed (Project Roadmap)

| Step  | What                     | How                                                                                                | Code Location                             |
| ----- | ------------------------ | -------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **1** | Run this exploration     | `python transformerLenstest.py`                                                                    | transformerLenstest.py                    |
| **2** | Scale to full dataset    | Load CounterFact with `load_counterfact()`, loop through all entries                               | load_dataset.py                           |
| **3** | Compute SGR distribution | Compute SGR for every prompt pair, plot histogram                                                  | New file: `src/benchmark/sgr_analysis.py` |
| **4** | Per-head decomposition   | Use `cache.decompose_resid(mode="attn")` at the **head** level to find specific "Inhibition Heads" | Extend Phase 4                            |
| **5** | Add Pythia-160m          | Just change `model_name` to `"pythia-160m"` — your `load_model()` already supports it              | load_models.py                            |
| **6** | Artificial amplification | Hook into inhibition heads and multiply their outputs by 2×, 3× to test if negation can be "fixed" | New experiment                            |
| **7** | Statistical analysis     | Correlate SGR with negation failure rate, compute p-values                                         | metrics.py                                |

You can run the file now to see all the outputs. The existing src modules you've already built (model loading, dataset loading, prompt building, benchmarking) slot in perfectly — this exploration file shows you what each module needs to produce and consume.

Completed (3/3) _Explain project roadmap_

Made changes.
