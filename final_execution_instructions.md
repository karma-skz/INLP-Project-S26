# Final Execution Guide (Comprehensive Scaled Edition)

This guide documents the final run commands necessary to execute the entire benchmarking and analysis pipeline across the original baseline models PLUS the larger parametrised models (`gpt2-medium`, `gpt2-large`, `pythia-410m`). 

*(Warning: Because the large models contain significantly more layers and thicker hidden dimensions, the computational timescale is heavily expanded).*

---

### Step 1: Regenerate The Core Baseline Matrix
*(This must be run first before Step 2)*

Command:
```bash
conda run -n inlp-project python run_pipeline.py --models gpt2-small pythia-160m gpt2-medium gpt2-large pythia-410m --max_samples -1
```

* **Estimated Time:** 4 to 6 Hours
* **What it does:** Iterates over all 5 models without bounds, testing all ~20,000 entities in CounterFact. This constructs massive `results/*.csv` datasets with cross-model metrics which all downstream offline tools rely on.

---

### Step 2: Instant Offline Analysis
*(Can only be run once Step 1 is finished)*

Commands:
```bash
conda run -n inlp-project python sgr_lt1_verification.py
conda run -n inlp-project python hard_negation_audit.py
conda run -n inlp-project python crossover_analysis.py
```

* **Estimated Time:** Less than 1 minute (Combined)
* **What it does:** These are pure Data-Science modules. They ingest all 5 of your massive benchmark datasets and seamlessly overlay Pythia-410m and GPT-2 Large onto the scatterplots/box-plots to test generalisation.

---

### Step 3: Finer Cross-Model Extended Amplification
*(Can be run totally independently of Step 1 and 2)*

Command:
```bash
conda run -n inlp-project python extended_amplification.py --models gpt2-small pythia-160m gpt2-medium gpt2-large pythia-410m --max_samples 1000
```

* **Estimated Time:** 3 to 4 Hours
* **What it does:** Iterates the logic amplification multipliers across all 5 models. We cap at 1,000 samples here to make the timeline viable while cleanly smoothing out the line graphs comparing `gpt2-small` against `gpt2-large`.

---

### Step 4: Causal Activation Flow (Patching) Tracker
*(Can be run totally independently of all other steps)*

Command:
```bash
conda run -n inlp-project python activation_patching.py --models gpt2-small pythia-160m gpt2-medium gpt2-large pythia-410m --max_samples 1000
```

* **Estimated Time:** ~6 to 8 Hours (Run Overnight)
* **What it does:** The most unoptimized runtime task. Because `gpt2-large` contains 36 layers, caching and injecting the 3 geometry flows perfectly takes O(36 x 3 x Failures) passes. Testing this horizontally across 5 models requires huge pipeline saturation. Capping the search bounds to `1000` samples per model guarantees it will finish while revealing whether the layer-5 "Causal Rescue Range" shifts outward in thicker models.
