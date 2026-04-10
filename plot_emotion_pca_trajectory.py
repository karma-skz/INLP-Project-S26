"""
plot_emotion_pca_trajectory.py
================================
Plot the average layer-wise PCA trajectory of negated joy in Pythia-160M.

Steps:
  1. Load Pythia-160M via TransformerLens.
  2. Build four prompt sets: Target (joy+positive), Antonym (sadness+positive),
     Neutral (neutral+positive), Negated Target (joy+negated).
  3. Extract final-layer resid_post for Target/Antonym/Neutral anchors and
     fit a PCA(2) on those representations.
  4. Extract all-layer resid_post for Negated Target prompts.
  5. Transform everything through PCA, compute centroids, and plot.

Usage
-----
    conda activate inlp-project
    python plot_emotion_pca_trajectory.py
"""

from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── Monkey-patch for Pythia loading ──────────────────────────────────────────
try:
    from transformers import GPTNeoXConfig
    if not hasattr(GPTNeoXConfig, "rotary_pct"):
        GPTNeoXConfig.rotary_pct = 0.25
except ImportError:
    pass

import transformer_lens

# ── Data ─────────────────────────────────────────────────────────────────────

PAIR_TEMPLATES: list[tuple[str, str, str]] = [
    ("plain_feel",  "I feel {lexeme}",                  "I do not feel {lexeme}"),
    ("right_now",   "Right now I feel {lexeme}",        "Right now I do not feel {lexeme}"),
    ("today",       "Today I feel {lexeme}",            "Today I do not feel {lexeme}"),
    ("moment",      "At the moment I feel {lexeme}",    "At the moment I do not feel {lexeme}"),
    ("cause",       "This makes me feel {lexeme}",      "This does not make me feel {lexeme}"),
    ("lately",      "Lately I have felt {lexeme}",      "Lately I have not felt {lexeme}"),
]

DEFAULT_EMOTION_LEXEMES: dict[str, list[str]] = {
    "joy":     ["happy", "joyful", "glad", "cheerful"],
    "sadness": ["sad", "unhappy", "gloomy", "miserable"],
}

DEFAULT_NEUTRAL_LEXEMES: list[str] = ["calm", "okay", "neutral", "steady"]

MODELS = ["gpt2-small", "pythia-160m"]
FIG_DIR    = "figures/emotion_negation"


# ── Prompt generation ────────────────────────────────────────────────────────

def _positive_prompts(lexemes: list[str]) -> list[str]:
    prompts = []
    for _, pos_tmpl, _ in PAIR_TEMPLATES:
        for lex in lexemes:
            prompts.append(pos_tmpl.format(lexeme=lex))
    return prompts

def _negated_prompts(lexemes: list[str]) -> list[str]:
    prompts = []
    for _, _, neg_tmpl in PAIR_TEMPLATES:
        for lex in lexemes:
            prompts.append(neg_tmpl.format(lexeme=lex))
    return prompts


# ── Extraction helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_final_layer_resid(model, prompts: list[str]) -> np.ndarray:
    n_layers = model.cfg.n_layers
    hook_name = f"blocks.{n_layers - 1}.hook_resid_post"
    vecs = []
    for prompt in prompts:
        _, cache = model.run_with_cache(prompt, names_filter=[hook_name])
        vec = cache[hook_name][0, -1, :].cpu().numpy()
        vecs.append(vec)
    return np.stack(vecs)

@torch.no_grad()
def extract_all_layers_resid(model, prompts: list[str]) -> np.ndarray:
    n_layers = model.cfg.n_layers
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    all_vecs = []
    for prompt in prompts:
        _, cache = model.run_with_cache(prompt, names_filter=hook_names)
        layer_vecs = []
        for l in range(n_layers):
            vec = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu().numpy()
            layer_vecs.append(vec)
        all_vecs.append(np.stack(layer_vecs))
    return np.stack(all_vecs)

def plot_single_model_3_subplots(model_name, n_layers, num_prompts, target_2d, antonym_2d, neutral_2d,
                                 target_centroid, antonym_centroid, neutral_centroid, negated_means):
    print(f"Plotting 3-subplot graph for {model_name}...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 24))

    for ax in [ax1, ax2, ax3]:
        ax.scatter(target_2d[:, 0], target_2d[:, 1], c="mediumseagreen", alpha=0.25, s=30, edgecolors="none")
        ax.scatter(antonym_2d[:, 0], antonym_2d[:, 1], c="indianred", alpha=0.25, s=30, edgecolors="none")
        ax.scatter(neutral_2d[:, 0], neutral_2d[:, 1], c="silver", alpha=0.25, s=30, edgecolors="none")
        ax.scatter(*target_centroid, c="green", s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=5, label="Joy (target)")
        ax.scatter(*antonym_centroid, c="red", s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=5, label="Sadness (antonym)")
        ax.scatter(*neutral_centroid, c="gray", s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=5, label="Neutral")
        ax.annotate("Joy", target_centroid, fontsize=11, fontweight="bold", color="green", xytext=(8, 8), textcoords="offset points")
        ax.annotate("Sadness", antonym_centroid, fontsize=11, fontweight="bold", color="red", xytext=(8, 8), textcoords="offset points")
        ax.annotate("Neutral", neutral_centroid, fontsize=11, fontweight="bold", color="gray", xytext=(8, 8), textcoords="offset points")

    # ax1: Full
    ax1.plot(negated_means[:, 0], negated_means[:, 1], color="darkorange", linewidth=2.5, alpha=0.85, zorder=4, label="Negated Joy trajectory")
    ax1.scatter(negated_means[:, 0], negated_means[:, 1], c=range(n_layers), cmap="YlOrRd", s=70, edgecolors="black", linewidths=0.6, zorder=5)
    for l in [0, n_layers // 2, n_layers - 1]:
        ax1.annotate(f"L{l}", negated_means[l], fontsize=9, fontweight="bold", color="darkorange",
                    xytext=(10, -12 if l == n_layers - 1 else 10), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))
    ax1.set_title("Full Trajectory (L0 to L11)", fontsize=13, fontweight="bold")

    # ax2: Zoomed L0-L10
    zoomed_means = negated_means[:11]
    ax2.plot(zoomed_means[:, 0], zoomed_means[:, 1], color="darkorange", linewidth=2.5, alpha=0.85, zorder=4, label="Negated Joy trajectory (L0-L10)")
    ax2.scatter(zoomed_means[:, 0], zoomed_means[:, 1], c=range(11), cmap="YlOrRd", s=70, edgecolors="black", linewidths=0.6, zorder=5)
    for l in range(11):
        ax2.annotate(f"L{l}", zoomed_means[l], fontsize=9, fontweight="bold", color="darkorange", xytext=(8, 8), textcoords="offset points")
    x_min, x_max = zoomed_means[:, 0].min(), zoomed_means[:, 0].max()
    ax2.set_xticks(np.arange(np.floor(x_min * 4) / 4, np.ceil(x_max * 4) / 4 + 0.25, 0.25))
    ax2.set_title("Zoomed Trajectory (L0 to L10)", fontsize=13, fontweight="bold")

    # ax3: Super Zoomed L0-L6
    super_zoomed_means = negated_means[:7]
    ax3.plot(super_zoomed_means[:, 0], super_zoomed_means[:, 1], color="darkorange", linewidth=2.5, alpha=0.85, zorder=4, label="Negated Joy trajectory (L0-L6)")
    ax3.scatter(super_zoomed_means[:, 0], super_zoomed_means[:, 1], c=range(7), cmap="YlOrRd", s=70, edgecolors="black", linewidths=0.6, zorder=5)
    for l in range(7):
        ax3.annotate(f"L{l}", super_zoomed_means[l], fontsize=9, fontweight="bold", color="darkorange", xytext=(8, 8), textcoords="offset points")
    x3_min, x3_max = super_zoomed_means[:, 0].min(), super_zoomed_means[:, 0].max()
    ax3.set_xticks(np.arange(np.floor(x3_min * 20) / 20, np.ceil(x3_max * 20) / 20 + 0.05, 0.05))
    ax3.set_title("Zoomed Trajectory (L0 to L6)", fontsize=13, fontweight="bold")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_facecolor("#fafafa")
        if ax == ax1:
            ax.legend(fontsize=10, loc="best")

    fig.suptitle(f"Average Layer-wise Trajectory of Negated Joy in {model_name}\n(PCA fit on final-layer anchors; {num_prompts} negated prompts)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(FIG_DIR, f"pca_negation_trajectory_3subplots_{model_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    target_prompts  = _positive_prompts(DEFAULT_EMOTION_LEXEMES["joy"])
    antonym_prompts = _positive_prompts(DEFAULT_EMOTION_LEXEMES["sadness"])
    neutral_prompts = _positive_prompts(DEFAULT_NEUTRAL_LEXEMES)
    negated_prompts = _negated_prompts(DEFAULT_EMOTION_LEXEMES["joy"])

    all_model_data = {}

    for model_name in MODELS:
        print(f"\n================ Loading {model_name} ================")
        model = transformer_lens.HookedTransformer.from_pretrained(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        n_layers = model.cfg.n_layers

        target_vecs  = extract_final_layer_resid(model, target_prompts)
        antonym_vecs = extract_final_layer_resid(model, antonym_prompts)
        neutral_vecs = extract_final_layer_resid(model, neutral_prompts)

        combined = np.vstack([target_vecs, antonym_vecs, neutral_vecs])
        pca = PCA(n_components=2)
        pca.fit(combined)

        target_2d  = pca.transform(target_vecs)
        antonym_2d = pca.transform(antonym_vecs)
        neutral_2d = pca.transform(neutral_vecs)

        target_centroid  = target_2d.mean(axis=0)
        antonym_centroid = antonym_2d.mean(axis=0)
        neutral_centroid = neutral_2d.mean(axis=0)

        print("\nAnchor Centroid Coordinates (Final Layer PCA space):")
        print(f"  Joy (Target):    X = {target_centroid[0]:8.4f}, Y = {target_centroid[1]:8.4f}")
        print(f"  Sadness (Ant.):  X = {antonym_centroid[0]:8.4f}, Y = {antonym_centroid[1]:8.4f}")
        print(f"  Neutral:         X = {neutral_centroid[0]:8.4f}, Y = {neutral_centroid[1]:8.4f}")

        negated_all = extract_all_layers_resid(model, negated_prompts)
        negated_means = []
        for l in range(n_layers):
            layer_vecs = negated_all[:, l, :]
            layer_2d = pca.transform(layer_vecs)
            negated_means.append(layer_2d.mean(axis=0))
        negated_means = np.array(negated_means)

        print("\nNegated Joy Trajectory Coordinates per Layer:")
        for l in range(n_layers):
            print(f"  Layer {l:2d}: X = {negated_means[l][0]:8.4f}, Y = {negated_means[l][1]:8.4f}")

        # Store for full side-by-side plot
        all_model_data[model_name] = {
            "n_layers": n_layers, "target_2d": target_2d, "antonym_2d": antonym_2d, "neutral_2d": neutral_2d,
            "target_centroid": target_centroid, "antonym_centroid": antonym_centroid, "neutral_centroid": neutral_centroid,
            "negated_means": negated_means
        }

        # Plot 3-subplot graph for this model
        plot_single_model_3_subplots(model_name, n_layers, len(negated_prompts), target_2d, antonym_2d, neutral_2d,
                                     target_centroid, antonym_centroid, neutral_centroid, negated_means)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Side-by-Side Comparison Plot ---
    print("\nPlotting side-by-side normal trajectory comparison...")
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 7))
    for ax, model_name in zip(axes, MODELS):
        data = all_model_data[model_name]
        n_layers = data["n_layers"]
        
        ax.scatter(data["target_2d"][:, 0], data["target_2d"][:, 1], c="mediumseagreen", alpha=0.25, s=30, edgecolors="none")
        ax.scatter(data["antonym_2d"][:, 0], data["antonym_2d"][:, 1], c="indianred", alpha=0.25, s=30, edgecolors="none")
        ax.scatter(data["neutral_2d"][:, 0], data["neutral_2d"][:, 1], c="silver", alpha=0.25, s=30, edgecolors="none")

        ax.scatter(*data["target_centroid"], c="green", s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=5, label="Joy (target)")
        ax.scatter(*data["antonym_centroid"], c="red", s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=5, label="Sadness (antonym)")
        ax.scatter(*data["neutral_centroid"], c="gray", s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=5, label="Neutral")

        ax.annotate("Joy", data["target_centroid"], fontsize=11, fontweight="bold", color="green", xytext=(8, 8), textcoords="offset points")
        ax.annotate("Sadness", data["antonym_centroid"], fontsize=11, fontweight="bold", color="red", xytext=(8, 8), textcoords="offset points")
        ax.annotate("Neutral", data["neutral_centroid"], fontsize=11, fontweight="bold", color="gray", xytext=(8, 8), textcoords="offset points")

        negated = data["negated_means"]
        ax.plot(negated[:, 0], negated[:, 1], color="darkorange", linewidth=2.5, alpha=0.85, zorder=4, label="Negated Trajectory")
        ax.scatter(negated[:, 0], negated[:, 1], c=range(n_layers), cmap="YlOrRd", s=70, edgecolors="black", linewidths=0.6, zorder=5)

        for l in [0, n_layers // 2, n_layers - 1]:
            ax.annotate(f"L{l}", negated[l], fontsize=9, fontweight="bold", color="darkorange",
                        xytext=(10, -12 if l == n_layers - 1 else 10), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))

        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)
        ax.set_title(f"{model_name.upper()} (L0 to L{n_layers-1})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_facecolor("#fafafa")

    fig.suptitle(f"Cross-Model Comparison: Average Layer-wise Trajectory of Negated Joy", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path_comp = os.path.join(FIG_DIR, "pca_negation_trajectory_comparison.png")
    fig.savefig(out_path_comp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison: {out_path_comp}")
    print("Done!")

if __name__ == "__main__":
    import gc
    main()
