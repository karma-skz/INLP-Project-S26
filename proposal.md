# Research Proposal: Deciphering the Compositional Collapse

**Subtitle:** A Mechanistic Interpretability Study of Negation and Logical Parity in Large Language Models

## 1. Overview & Motivation
Negation is a fundamental syntactic operation that forces language models to dynamically invert heavily weighted pre-training priors. While significant behavioral research tracks how Large Language Models (LLMs) hallucinate under single negation ($\neg P$), the underlying mechanistic architecture remains poorly understood. Furthermore, when logical complexity scales to compositional double negation ($\neg\neg P \equiv P$), models experience a profound "compositional collapse". 

This project maps the mechanistic architecture of **Negation as a whole**—from the foundational "Logic-Memory Conflict" of single negation to the feature-level interference of double negation—across frontier models.

## 2. Publishability & Novelty Factors
This project targets top-tier venues (ICLR, NeurIPS) by bridging a critical gap between theoretical architecture and empirical alignment:
- **The Competitive Gating Hypothesis:** We introduce a novel framing of negation not as a state, but as a physical, measurable conflict in the residual stream between early-layer FFN memory retrieval (the signal) and late-layer Attention suppression (the gate).
- **The Soft Negation Paradox:** We empirically challenge linguistic assumptions by proving that "soft" probabilistic negators (e.g., "rarely") paradoxically *reduce* hallucinations compared to explicit hard negators (e.g., "not"), owing to a measurable weakening of the FFN retrieval signal.
- **Latent Compositionality:** No existing literature has mapped the causal circuitry of Double Negation Elimination. We introduce feature-level causal tracing using Sparse Autoencoders (SAEs) to define exactly how sequential logical operators interact (and destructively interfere) in latent space.

## 3. Research Questions
- **The Logic-Memory Conflict (Single Negation):** At exactly what layer does logical syntax (Attention) attempt to overpower factual recall (FFN)? How does the Signal-to-Gate Ratio (SGR) predict hallucination rates?
- **Compositional Interference (Double Negation):** How do two sequential negation operators interact in latent space? Do they destructively interfere, or does the second negator simply fail to activate a secondary inhibition circuit?
- **The Excluded Middle:** By tracking Latent Truth-Value, can we pinpoint the exact transformer layer where the model state becomes trapped between $P$ and $\neg P$?

## 4. Methodology & Goals
Our experimental methodology is divided into three distinct phases, mapped directly to our interpretability pipelines, to systematically prove the causes of negation collapse.

### Phase 1: Proving the Logic-Memory Conflict (Single Negation)
**Goal:** Establish the baseline physical mechanism of single negation failure.
**Methodology:** 
We calculate the **Signal-to-Gate Ratio (SGR)** and utilize per-head **Direct Logit Attribution (DLA)** across Transformer layers. We hypothesize that FFNs acting in early/middle layers retrieve the factual answer (the "Signal"), while late-layer Attention heads act as suppressors (the "Gate"). When the factual signal mathematically overpowers the attention gate in the residual stream, the model hallucinates the affirmative answer.

### Phase 2: Mapping the "Excluded Middle" via Latent Probes
**Goal:** Identify the exact layer where the model state collapses during Double Negation ($\neg\neg P$).
**Methodology:** 
We train `scikit-learn` Logistic Regression probes on the residual stream activations of $P$ and $\neg P$ contexts. We then pass the double-negated prompt ($\neg\neg P$) through the model and calculate the **Latent Truth-Value Toggling (LTVT)** at each layer. This tracks the probability mass shifting between the True and False states, pinpointing exactly where the logic fails to flip back to $P$ and instead gets trapped in an ambiguous "Excluded Middle."

### Phase 3: Causal Tracing of Compositional Interference
**Goal:** Prove that the secondary negator actively destroys the primary negator's circuit, rather than simply failing to trigger its own.
**Methodology:** 
Using `TransformerLens`, we perform targeted causal interventions (Activation Patching and Feature Steering). We compute the novel **Compositional Interference Score (CIS)**, which measures the difference in DLA of a known inhibition head under single versus double negation contexts. A CIS score $> 1.0$ mathematically proves that the presence of the second negator actively attenuates the intervention effect of the first negator's circuit (destructive interference).

### Phase 4: Feature-Level Semantic Tracking
**Goal:** Map the human-interpretable features responsible for negation logic.
**Methodology:** 
We integrate the `sae_lens` library to extract sparse features from the dense hidden states across layers. By tracking **SAE Feature Trajectories**, we can observe if a universal "Negation Operator" feature exists, and if it fails to activate twice in sequence during Double Negation Elimination.

## 5. Datasets
To evaluate the mechanistic failure modes of negation across both synthetic isolation and natural downstream contexts, we employ a diverse benchmark suite:
- **Negation-Parity-Bench:** A rigorously constructed dataset of 10,000+ triplets ($P$, $\neg P$, $\neg\neg P$) spanning Factual Parity and Semantic Parity.
- **Negated LAMA:** For evaluating the factual knowledge collapse under single and double negation.
- **CondaQA:** A natural question-answering corpus to test how mechanistic logic failures disrupt reasoning chains.
- **Thunder-NUBench:** A comprehensive sentence-level negation benchmark used for evaluating out-of-distribution parity capabilities.
- **Reasoning Parity:** Modified math datasets (like GSM8K) with negated constraints to test downstream logical inversion.

## 6. Target Models
- **Primary Analysis:** `Llama-3-8B` and `Gemma-2-9B`
- **Scale Validation:** `Llama-3-70B` and `Gemma-2-27B`
