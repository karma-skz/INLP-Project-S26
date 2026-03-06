# Data

This directory stores downloaded and processed datasets.

## CounterFact Dataset

The project uses the [NeelNanda/counterfact-tracing](https://huggingface.co/datasets/NeelNanda/counterfact-tracing) dataset from HuggingFace.

The dataset is automatically downloaded on first run via the `src/dataset/load_dataset.py` module.

## Contents

After running the benchmark, this directory may contain cached dataset files.
These are excluded from version control via `.gitignore`.
