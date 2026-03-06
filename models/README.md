# Models

This directory is reserved for cached model weights.

## Supported Models

- **GPT-2 Small** (`gpt2-small`) — 124M parameters
- **Pythia-160M** (`pythia-160m`) — 160M parameters

Models are loaded via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
and cached automatically on first download.

Model files are excluded from version control via `.gitignore`.
