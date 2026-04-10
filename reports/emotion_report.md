In our study of emotional negation within small Language Models (LLMs), we analyze the latent representations of GPT2-small (124M) and Pythia-160m. Our observations provide a window into how these models geometrically encode affect and how they struggle with the logical operator of negation.

Here is our explanation of the findings, supported by research in mechanistic interpretability and NLP.

### 1. Late-Layer Peak Direction Strength
**The Observation:** In GPT2-small, we observed that the `peak_direction_norm` for all emotions peaks at the final layer (Layer 11), with high values ranging from 70.6 to 79.2. In Pythia-160m, the peak occurs slightly earlier for some emotions (Layer 9 for anger, fear, and sadness) with significantly lower norms (~8.4 to 10.5).

**Our Explanation:** We find that emotional features are high-level semantic abstractions that require the full depth of the transformer to materialize. Research by **Geva et al. (2020)** suggests that Transformer Feed-Forward Networks (FFNs) act as key-value memories, where lower layers handle surface-level syntax and higher layers resolve complex semantic concepts. The fact that GPT2-small peaks at the very last layer suggests it is utilizing its final residual updates to "solidify" the emotional concept before the unembedding head. The disparity in norms between the two models likely reflects differences in weight initialization and internal scaling (LayerNorm placement), as Pythia models often exhibit different activation variances compared to the GPT2 family (**Biderman et al., 2023**).

### 2. Residual Stream Expansion (Early vs. Late Representation)
**The Observation:** In both models, we see a massive increase in the representation norm from early to late layers. For GPT2-small, the norm jumps from ~13 (early) to ~50 (late). Pythia-160m shows a similar doubling from ~4 to ~8.

**Our Explanation:** This follows the "Logit Lens" and "Residual Stream as a Communication Channel" framework described by **Elhage et al. (2021)**. In the early layers, the model is still processing token positions and basic grammar; the "emotional direction" is not yet clear, leading to a lower norm. As information passes through successive attention and MLP blocks, the model accumulates evidence for the emotion, "writing" more information into the residual stream. The late-layer norm is higher because the model has reached a point of semantic saturation where it is preparing to predict the next token.

### 3. The Negation Distance Gap and Logic Failures
**The Observation:** In the `negation_distance_gap` (distance to opposite - distance to neutral), a negative value means the negated prompt (e.g., "I am not happy") is closer to the opposite emotion ("sad") than to a neutral state. We observed that GPT2-small shows negative gaps for Joy (-20.6) and Sadness (-7.5). However, Pythia-160m shows a positive gap for Sadness (0.88), meaning "not sad" is actually closer to a neutral state than to a "joyful" state.

**Our Explanation:** Negation is a notorious weakness for small LLMs. Research by **Kassner and Schütze (2020)** in *"Negated and Misprimed Probes"* demonstrates that LLMs often treat negated prompts similarly to their affirmative counterparts, focusing on the "topic" (the emotion) rather than the "logical flip." 
*   In GPT2, the negative gap suggests the model is performing a crude "semantic inversion"—it sees "not" and "joy" and moves the vector toward the opposite pole. 
*   In Pythia, the positive gap for sadness suggests the model is failing to find an "opposite" and instead collapses the representation back toward a generic, neutral state. This indicates that Pythia-160m may have a more "unipolar" representation of certain emotions compared to GPT2.

### 4. Closer to Neutral Rate
**The Observation:** In GPT2-small, the `closer_to_neutral_rate` for Joy is 0.0 and Sadness is 0.20. In Pythia-160m, these rates are higher: Joy (0.22) and Sadness (0.77).

**Our Explanation:** A rate of 0.0 for Joy in GPT2-small means that in *every* test case, the negated joy prompt was further from neutral (and closer to the opposite). A rate of 0.77 for Sadness in Pythia means the model almost always views "not sad" as "neutral." This reflects the **"Excluded Middle" fallacy** in small models. As noted by **Hossain et al. (2020)**, small models struggle to distinguish between *contraries* (happy/sad) and *contradictories* (happy/not-happy). Pythia’s high rate for sadness suggests it lacks the nuanced latent space to map "not sad" to a positive affect, defaulting instead to a null-sentiment state.

### 5. Intervention Linearity and R² (The Directional Hypothesis)
**The Observation:** 
*   **GPT2-small:** Anger, Fear, and Sadness show very high linearity ($R^2 \approx 0.90–0.97$), but Joy has a very low $R^2$ of 0.22. 
*   **Pythia-160m:** Anger shows low linearity ($R^2 = 0.28$), while Fear, Joy, and Sadness are highly linear ($R^2 > 0.87$).

**Our Explanation:** The "Linear Representation Hypothesis" (**Park et al., 2023**) posits that concepts are represented as directions in LLM latent space. 
*   High $R^2$ values indicate that we can "steer" the model's emotion by simply adding a constant vector. 
*   The anomaly in GPT2-small (Joy) and Pythia-160m (Anger) suggests these specific emotions are not encoded as a single linear "axis." Instead, they might be represented as a **multi-modal distribution** or a non-linear manifold. For instance, in GPT2-small, "Joy" might be so entangled with other positive sentiments that a single linear intervention fails to capture its variance, leading to the low $R^2$. This aligns with findings by **Li et al. (2023)** that while many concepts are linear, highly frequent or "polysemous" concepts often require non-linear probes to extract accurately.

### 6. Intervention Slopes
**The Observation:** Most slopes are near zero (e.g., GPT2 Fear: -0.00014; Pythia Joy: 0.00036). 

**Our Explanation:** The extremely small magnitude of the slopes indicates that while the relationship is linear (high $R^2$), the "sensitivity" of the model’s logits to the latent emotional direction is low. This suggests that the emotional "direction" we identified is orthogonal to the model's primary decision-making pathways for next-token prediction in these specific prompts. Even though we can identify the "direction" of anger or joy, small models like these require significant "force" (larger intervention coefficients) to actually change their output behavior, a phenomenon documented in **"Activation Addition" (Rimsky et al., 2023)** research.