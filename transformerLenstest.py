"""
=============================================================================
Pink Elephants — Full Pipeline Exploration (Single Prompt Pair)
=============================================================================
This script walks through EVERY technique described in the project proposal
using one concrete example:

    Positive prompt : "The capital of France is"       → target: " Paris"
    Negated prompt  : "The capital of France is not"   → target should NOT be " Paris"

Pipeline overview
-----------------
  Phase 0 — Setup & sanity check
  Phase 1 — Behavioural comparison (logits, top-k predictions)
  Phase 2 — Cache decomposition of the residual stream
  Phase 3 — Direct Logit Attribution (DLA) per layer
  Phase 4 — Identifying Memory Layers (FFNs) vs Inhibition Heads (Attn)
  Phase 5 — Signal-to-Gate Ratio (SGR)
  Phase 6 — Activation Patching (causal intervention)
  Phase 7 — Visualization

Each section prints clear outputs so you can verify everything works.
"""

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend so it works headless
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from functools import partial

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(67)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Setup & Sanity Check
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 0 — Loading model & defining prompts")
print("=" * 72)

model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

POSITIVE_PROMPT = "The capital of France is"
NEGATED_PROMPT = "The capital of France is not"
TARGET_TOKEN = " Paris"  # note: leading space (GPT-2 tokenizer convention)

# Convert target to a single token id
target_id = model.to_single_token(TARGET_TOKEN)
print(f"Positive prompt : '{POSITIVE_PROMPT}'")
print(f"Negated prompt  : '{NEGATED_PROMPT}'")
print(f"Target token    : '{TARGET_TOKEN}' (id={target_id})")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Behavioural Comparison
# ═══════════════════════════════════════════════════════════════════════════════
# Here we simply run the model on both prompts and look at the final
# prediction distribution.  This is the "surface-level" check: does the
# model still predict "Paris" even when the prompt says "is not"?
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 1 — Behavioural comparison")
print("=" * 72)


def get_logits_and_top_k(prompt: str, k: int = 10):
    """Run the model and return last-position logits + top-k tokens."""
    tokens = model.to_tokens(prompt)
    logits = model(tokens)  # (batch, seq, vocab)
    last_logits = logits[0, -1, :]  # logits at the *next-token* position
    probs = torch.softmax(last_logits, dim=-1)
    top_k = torch.topk(probs, k)
    top_tokens = [model.to_string(t.item()) for t in top_k.indices]
    top_probs = top_k.values.tolist()
    target_logit = last_logits[target_id].item()
    target_prob = probs[target_id].item()
    target_rank = (last_logits >= last_logits[target_id]).sum().item()
    return {
        "logits": last_logits,
        "target_logit": target_logit,
        "target_prob": target_prob,
        "target_rank": target_rank,
        "top_tokens": top_tokens,
        "top_probs": top_probs,
    }


with torch.no_grad():
    pos_info = get_logits_and_top_k(POSITIVE_PROMPT)
    neg_info = get_logits_and_top_k(NEGATED_PROMPT)

print(f"\n{'':>30}  {'Positive':>12}  {'Negated':>12}")
print(
    f"{'Target logit':>30}  {pos_info['target_logit']:>12.3f}  {neg_info['target_logit']:>12.3f}"
)
print(
    f"{'Target prob':>30}  {pos_info['target_prob']:>12.4f}  {neg_info['target_prob']:>12.4f}"
)
print(
    f"{'Target rank':>30}  {pos_info['target_rank']:>12d}  {neg_info['target_rank']:>12d}"
)

print(
    f"\nTop-5 predictions (positive): ",
    list(
        zip(pos_info["top_tokens"][:5], [f"{p:.4f}" for p in pos_info["top_probs"][:5]])
    ),
)
print(
    f"Top-5 predictions (negated) : ",
    list(
        zip(neg_info["top_tokens"][:5], [f"{p:.4f}" for p in neg_info["top_probs"][:5]])
    ),
)

# Negation failure: the negated prompt should INCREASE the target's rank
# (lower probability). If the rank is lower (= higher prob) in the negated
# prompt than in the positive prompt, the model failed to suppress the target.
negation_failure = neg_info["target_rank"] < pos_info["target_rank"]
print(
    f"\n→ Negation failure? {'YES — target ranked higher after negation!' if negation_failure else 'No — model suppressed it.'}"
)
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Cache Decomposition of the Residual Stream
# ═══════════════════════════════════════════════════════════════════════════════
# The residual stream is the "highway" that runs through the transformer.
# Every layer (attention + FFN) *adds* its output to it.  TransformerLens
# lets us decompose the final residual stream into the individual
# contributions of each component.  This tells us "who wrote what" into
# the stream.
#
# We decompose in two modes:
#   • mode="attn" — shows embed + each attn layer's contribution
#   • mode="full" — shows embed + attn + mlp for every layer
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 2 — Residual stream decomposition")
print("=" * 72)

with torch.no_grad():
    pos_logits, pos_cache = model.run_with_cache(POSITIVE_PROMPT)
    neg_logits, neg_cache = model.run_with_cache(NEGATED_PROMPT)

# Decompose residual stream into per-component contributions
# Shape: (n_components, batch, seq_len, d_model)
pos_resid, pos_labels = pos_cache.decompose_resid(return_labels=True, mode="full")
neg_resid, neg_labels = neg_cache.decompose_resid(return_labels=True, mode="full")

print(f"Number of residual-stream components: {len(pos_labels)}")
print(f"Component names (first 10): {pos_labels[:10]}")
print(f"Shape of decomposed residual: {pos_resid.shape}")
print(f"  → (n_components, batch, seq_len, d_model)")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Direct Logit Attribution (DLA) Per Layer
# ═══════════════════════════════════════════════════════════════════════════════
# DLA is the *core* technique of this project.
#
# Idea: The final logit for token t is just the dot product of the
# residual stream with the *unembedding vector* for t:
#
#     logit(t) = residual_stream · W_U[:, t]
#
# Since the residual stream is a *sum* of all component outputs,
# we can attribute the final logit to each component:
#
#     logit(t) = Σ_c  component_c · W_U[:, t]
#
# A positive DLA means that component *pushed towards* predicting the
# target.  A negative DLA means it pushed *away* from it.
#
# For our project:
#   • FFN layers with large POSITIVE DLA = "memory retrieval" (they
#     inject the fact "capital of France → Paris")
#   • Attention heads with large NEGATIVE DLA in the negated run =
#     "inhibition heads" (they try to suppress "Paris")
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 3 — Direct Logit Attribution (DLA)")
print("=" * 72)

# Get the unembedding vector for the target token
# W_U shape: (d_model, vocab_size)
target_unembed = model.W_U[:, target_id]  # (d_model,)

# Compute DLA for each component at the LAST token position
# pos_resid shape: (n_components, batch=1, seq_len, d_model)
# We want the last seq position → [..., -1, :]
pos_dla = (pos_resid[:, 0, -1, :] @ target_unembed).detach().cpu().numpy()
neg_dla = (neg_resid[:, 0, -1, :] @ target_unembed).detach().cpu().numpy()

print(f"\n{'Component':<25} {'DLA (pos)':>12} {'DLA (neg)':>12} {'Δ (neg-pos)':>12}")
print("-" * 65)
for i, label in enumerate(pos_labels):
    delta = neg_dla[i] - pos_dla[i]
    # Only print components with meaningful contribution
    if abs(pos_dla[i]) > 0.3 or abs(neg_dla[i]) > 0.3 or abs(delta) > 0.3:
        print(f"{label:<25} {pos_dla[i]:>12.3f} {neg_dla[i]:>12.3f} {delta:>12.3f}")

print(f"\n{'SUM (sanity check)':<25} {pos_dla.sum():>12.3f} {neg_dla.sum():>12.3f}")
print(
    f"{'Actual target logit':<25} {pos_info['target_logit']:>12.3f} {neg_info['target_logit']:>12.3f}"
)
print("(These should roughly match — small discrepancy is from LayerNorm)")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Separating Memory (FFN) vs Inhibition (Attention)
# ═══════════════════════════════════════════════════════════════════════════════
# Now we split the DLA into two groups:
#   • FFN layers  → "memory / retrieval signal"
#   • Attn layers → "logic / inhibition signal"
#
# Key hypothesis: FFN contributions should be SIMILAR between positive
# and negated prompts (recall is "logic-blind").  Attention contributions
# should DIFFER (this is where negation processing happens).
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 4 — Memory (FFN) vs Inhibition (Attention) separation")
print("=" * 72)

ffn_mask = np.array(["mlp" in label.lower() for label in pos_labels])
attn_mask = np.array(["attn" in label.lower() for label in pos_labels])

ffn_pos_total = pos_dla[ffn_mask].sum()
ffn_neg_total = neg_dla[ffn_mask].sum()
attn_pos_total = pos_dla[attn_mask].sum()
attn_neg_total = neg_dla[attn_mask].sum()

print(f"\n{'Signal Source':<25} {'DLA Pos':>12} {'DLA Neg':>12} {'Δ':>12}")
print("-" * 65)
print(
    f"{'FFN (memory) total':<25} {ffn_pos_total:>12.3f} {ffn_neg_total:>12.3f} {ffn_neg_total - ffn_pos_total:>12.3f}"
)
print(
    f"{'Attn (logic) total':<25} {attn_pos_total:>12.3f} {attn_neg_total:>12.3f} {attn_neg_total - attn_pos_total:>12.3f}"
)

print(f"\n→ If FFN Δ ≈ 0, recall is 'logic-blind' (memory fires regardless).")
print(f"→ If Attn Δ is large & negative, attention heads try to suppress.")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Signal-to-Gate Ratio (SGR)
# ═══════════════════════════════════════════════════════════════════════════════
# Our novel metric:
#
#     SGR = |FFN contribution to target (retrieval)| /
#           |Attn contribution AGAINST target (inhibition)|
#
# SGR > 1 → memory overwhelms logic → hallucination / negation failure
# SGR < 1 → logic successfully suppresses → correct behaviour
#
# We compute this on the NEGATED prompt (that's where the conflict is).
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 5 — Signal-to-Gate Ratio (SGR)")
print("=" * 72)

# For the negated prompt:
#  • Retrieval signal = sum of POSITIVE DLA from FFN layers
#    (components that push *toward* the forbidden target)
#  • Inhibition signal = sum of NEGATIVE DLA from Attn layers
#    (components that push *away from* the forbidden target)

neg_ffn_dla = neg_dla[ffn_mask]
neg_attn_dla = neg_dla[attn_mask]

retrieval_strength = neg_ffn_dla[neg_ffn_dla > 0].sum()  # positive FFN pushes
inhibition_strength = abs(neg_attn_dla[neg_attn_dla < 0].sum())  # negative attn pushes

# Avoid division by zero
if inhibition_strength > 0:
    sgr = retrieval_strength / inhibition_strength
else:
    sgr = float("inf")

print(f"  Retrieval strength  (FFN → target)   : {retrieval_strength:.3f}")
print(f"  Inhibition strength (Attn ← target)  : {inhibition_strength:.3f}")
print(f"  Signal-to-Gate Ratio (SGR)            : {sgr:.3f}")
print()
if sgr > 1:
    print(f"  → SGR > 1: Memory OVERWHELMS logic. Negation failure expected!")
else:
    print(f"  → SGR < 1: Logic OVERRIDES memory. Correct suppression.")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5b — Per-Layer DLA breakdown (for finding the "Crossover Point")
# ═══════════════════════════════════════════════════════════════════════════════
# We look at how DLA accumulates layer by layer.  The "crossover point"
# is the layer where the cumulative inhibition signal starts to exceed
# the cumulative retrieval signal (or fails to).
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 5b — Cumulative DLA & Crossover Point")
print("=" * 72)

n_layers = model.cfg.n_layers
per_layer_ffn_dla_neg = np.zeros(n_layers)
per_layer_attn_dla_neg = np.zeros(n_layers)
per_layer_ffn_dla_pos = np.zeros(n_layers)
per_layer_attn_dla_pos = np.zeros(n_layers)

for i, label in enumerate(pos_labels):
    # Labels look like "0_attn_out", "0_mlp_out", "1_attn_out", etc.
    parts = label.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        layer_idx = int(parts[0])
        if "mlp" in label.lower():
            per_layer_ffn_dla_neg[layer_idx] += neg_dla[i]
            per_layer_ffn_dla_pos[layer_idx] += pos_dla[i]
        elif "attn" in label.lower():
            per_layer_attn_dla_neg[layer_idx] += neg_dla[i]
            per_layer_attn_dla_pos[layer_idx] += pos_dla[i]

cumulative_ffn = np.cumsum(per_layer_ffn_dla_neg)
cumulative_attn = np.cumsum(per_layer_attn_dla_neg)
cumulative_total = cumulative_ffn + cumulative_attn

print(
    f"\n{'Layer':<8} {'FFN DLA':>10} {'Attn DLA':>10} {'Cum FFN':>10} {'Cum Attn':>10} {'Cum Total':>10}"
)
print("-" * 62)
for layer in range(n_layers):
    print(
        f"{layer:<8} "
        f"{per_layer_ffn_dla_neg[layer]:>10.3f} "
        f"{per_layer_attn_dla_neg[layer]:>10.3f} "
        f"{cumulative_ffn[layer]:>10.3f} "
        f"{cumulative_attn[layer]:>10.3f} "
        f"{cumulative_total[layer]:>10.3f}"
    )

# Find crossover: the first layer where the cumulative total flips from
# negative (net inhibition) to positive (net retrieval / negation failure).
# If the total is always negative  → suppression holds throughout (success).
# If the total is always positive  → retrieval dominates from the start.
# If there is a sign change → crossover marks where retrieval wins.
crossover = None
for layer in range(1, n_layers):
    if cumulative_total[layer - 1] < 0 <= cumulative_total[layer]:
        crossover = layer
        break

if all(v >= 0 for v in cumulative_total):
    print(f"\n→ No crossover: retrieval dominates from the start. Negation fails.")
elif all(v < 0 for v in cumulative_total):
    print(f"\n→ No crossover: inhibition holds throughout. Successful suppression.")
else:
    print(f"\n→ Crossover at layer {crossover}: retrieval overcomes inhibition here.")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Activation Patching (Causal Intervention)
# ═══════════════════════════════════════════════════════════════════════════════
# Activation patching answers: "If I replace the activation at layer L
# (from the negated run) with the activation from the positive run,
# how much does the target logit change?"
#
# This identifies the CAUSAL importance of each layer for negation.
#
# Procedure:
#   1. Run the NEGATED prompt normally → get baseline target logit
#   2. For each layer, hook into the residual stream and REPLACE the
#      negated activation with the corresponding positive activation
#   3. Measure how the target logit changes
#
# If patching layer L causes the target logit to INCREASE a lot,
# that layer was doing important negation work (removing it lets
# the model "forget" the negation).
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 6 — Activation Patching")
print("=" * 72)

# First, get the clean activations from the positive prompt
pos_tokens = model.to_tokens(POSITIVE_PROMPT)
neg_tokens = model.to_tokens(NEGATED_PROMPT)

with torch.no_grad():
    # Run positive prompt and store all residual stream activations
    _, pos_cache_clean = model.run_with_cache(POSITIVE_PROMPT)
    # Baseline: negated prompt target logit
    neg_logits_baseline = model(neg_tokens)
    neg_target_logit_baseline = neg_logits_baseline[0, -1, target_id].item()

print(
    f"  Baseline target logit (negated, no patching): {neg_target_logit_baseline:.3f}"
)
print()


def patch_residual_hook(value, hook, pos_cache, layer_idx):
    """Hook that patches the residual stream at a specific layer.

    Because GPT-2 uses causal (unidirectional) attention, every position
    i sees only tokens 0..i. This means positions 0..(n-1) of the negated
    prompt have IDENTICAL residual-stream values to those same positions in
    the positive prompt — patching them would be a no-op and produce Δ=0.

    The only position that actually differs between the two prompts is the
    LAST position (where the model makes its next-token prediction). We
    therefore patch only position -1 of the negated run with position -1
    of the positive run, asking: "if this layer saw the positive-prompt
    prediction-position activation, how would the output change?"
    """
    pos_act = pos_cache[hook.name]
    # Patch only the last token position (prediction position)
    value[:, -1, :] = pos_act[:, -1, :]
    return value


patching_results_resid = {}

with torch.no_grad():
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Run with the patching hook
        patched_logits = model.run_with_hooks(
            NEGATED_PROMPT,
            fwd_hooks=[
                (
                    hook_name,
                    partial(
                        patch_residual_hook, pos_cache=pos_cache_clean, layer_idx=layer
                    ),
                )
            ],
        )
        patched_target_logit = patched_logits[0, -1, target_id].item()
        delta = patched_target_logit - neg_target_logit_baseline
        patching_results_resid[layer] = {
            "patched_logit": patched_target_logit,
            "delta": delta,
        }

print(
    f"{'Layer':<8} {'Patched logit':>14} {'Δ from baseline':>16} {'Interpretation':>30}"
)
print("-" * 72)
for layer in range(n_layers):
    r = patching_results_resid[layer]
    interp = ""
    if r["delta"] > 1.0:
        interp = "← negation processing here"
    elif r["delta"] < -1.0:
        interp = "← retrieval boosted here"
    print(f"{layer:<8} {r['patched_logit']:>14.3f} {r['delta']:>16.3f} {interp:>30}")

# Now do the same for MLP and Attention outputs specifically
print(f"\n--- Patching MLP outputs only ---")
patching_results_mlp = {}
with torch.no_grad():
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_mlp_out"
        patched_logits = model.run_with_hooks(
            NEGATED_PROMPT,
            fwd_hooks=[
                (
                    hook_name,
                    partial(
                        patch_residual_hook, pos_cache=pos_cache_clean, layer_idx=layer
                    ),
                )
            ],
        )
        patched_target_logit = patched_logits[0, -1, target_id].item()
        delta = patched_target_logit - neg_target_logit_baseline
        patching_results_mlp[layer] = delta

print(f"{'Layer':<8} {'Δ (MLP patch)':>14}")
print("-" * 25)
for layer in range(n_layers):
    marker = " ←" if abs(patching_results_mlp[layer]) > 0.5 else ""
    print(f"{layer:<8} {patching_results_mlp[layer]:>14.3f}{marker}")

print(f"\n--- Patching Attention outputs only ---")
patching_results_attn = {}
with torch.no_grad():
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_attn_out"
        patched_logits = model.run_with_hooks(
            NEGATED_PROMPT,
            fwd_hooks=[
                (
                    hook_name,
                    partial(
                        patch_residual_hook, pos_cache=pos_cache_clean, layer_idx=layer
                    ),
                )
            ],
        )
        patched_target_logit = patched_logits[0, -1, target_id].item()
        delta = patched_target_logit - neg_target_logit_baseline
        patching_results_attn[layer] = delta

print(f"{'Layer':<8} {'Δ (Attn patch)':>14}")
print("-" * 25)
for layer in range(n_layers):
    marker = " ←" if abs(patching_results_attn[layer]) > 0.5 else ""
    print(f"{layer:<8} {patching_results_attn[layer]:>14.3f}{marker}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Visualization
# ═══════════════════════════════════════════════════════════════════════════════
# Save a set of publication-quality figures summarizing the analysis.
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PHASE 7 — Generating figures")
print("=" * 72)

fig_dir = "figures"
import os

os.makedirs(fig_dir, exist_ok=True)

# ---- Figure 1: DLA comparison bar chart (positive vs negated) ----
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(n_layers)
width = 0.35
ax.bar(
    x - width / 2,
    per_layer_ffn_dla_pos,
    width,
    label="FFN (positive)",
    color="steelblue",
    alpha=0.8,
)
ax.bar(
    x + width / 2,
    per_layer_ffn_dla_neg,
    width,
    label="FFN (negated)",
    color="salmon",
    alpha=0.8,
)
ax.set_xlabel("Layer")
ax.set_ylabel("DLA contribution to target token")
ax.set_title(f"FFN (Memory) DLA — Positive vs Negated\nTarget: '{TARGET_TOKEN}'")
ax.legend()
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(x)
plt.tight_layout()
plt.savefig(f"{fig_dir}/01_ffn_dla_comparison.png", dpi=150)
print(f"  Saved {fig_dir}/01_ffn_dla_comparison.png")
plt.close()

# ---- Figure 2: Attention DLA comparison ----
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(
    x - width / 2,
    per_layer_attn_dla_pos,
    width,
    label="Attn (positive)",
    color="steelblue",
    alpha=0.8,
)
ax.bar(
    x + width / 2,
    per_layer_attn_dla_neg,
    width,
    label="Attn (negated)",
    color="salmon",
    alpha=0.8,
)
ax.set_xlabel("Layer")
ax.set_ylabel("DLA contribution to target token")
ax.set_title(f"Attention (Logic) DLA — Positive vs Negated\nTarget: '{TARGET_TOKEN}'")
ax.legend()
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(x)
plt.tight_layout()
plt.savefig(f"{fig_dir}/02_attn_dla_comparison.png", dpi=150)
print(f"  Saved {fig_dir}/02_attn_dla_comparison.png")
plt.close()

# ---- Figure 3: Cumulative DLA in the negated case ----
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    range(n_layers),
    cumulative_ffn,
    "o-",
    label="Cumulative FFN (retrieval)",
    color="crimson",
    linewidth=2,
)
ax.plot(
    range(n_layers),
    cumulative_attn,
    "s-",
    label="Cumulative Attn (inhibition)",
    color="dodgerblue",
    linewidth=2,
)
ax.plot(
    range(n_layers),
    cumulative_total,
    "^-",
    label="Cumulative Total",
    color="black",
    linewidth=2,
    linestyle="--",
)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
if crossover is not None:
    ax.axvline(
        x=crossover,
        color="green",
        linewidth=2,
        linestyle="--",
        label=f"Crossover @ layer {crossover}",
    )
ax.set_xlabel("Layer")
ax.set_ylabel("Cumulative DLA")
ax.set_title(
    f"Cumulative DLA (Negated Prompt) — Finding the Crossover Point\nTarget: '{TARGET_TOKEN}'"
)
ax.legend()
ax.set_xticks(range(n_layers))
plt.tight_layout()
plt.savefig(f"{fig_dir}/03_cumulative_dla_crossover.png", dpi=150)
print(f"  Saved {fig_dir}/03_cumulative_dla_crossover.png")
plt.close()

# ---- Figure 4: Activation patching results ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

layers = list(range(n_layers))

axes[0].bar(
    layers, [patching_results_resid[l]["delta"] for l in layers], color="mediumpurple"
)
axes[0].set_title("Residual Stream Patching")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Δ target logit")
axes[0].axhline(y=0, color="black", linewidth=0.5)

axes[1].bar(layers, [patching_results_mlp[l] for l in layers], color="salmon")
axes[1].set_title("MLP Output Patching")
axes[1].set_xlabel("Layer")
axes[1].axhline(y=0, color="black", linewidth=0.5)

axes[2].bar(layers, [patching_results_attn[l] for l in layers], color="steelblue")
axes[2].set_title("Attention Output Patching")
axes[2].set_xlabel("Layer")
axes[2].axhline(y=0, color="black", linewidth=0.5)

fig.suptitle(
    f"Activation Patching: Which layers carry negation information?\nTarget: '{TARGET_TOKEN}'",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(f"{fig_dir}/04_activation_patching.png", dpi=150)
print(f"  Saved {fig_dir}/04_activation_patching.png")
plt.close()

# ---- Figure 5: SGR visual ----
fig, ax = plt.subplots(figsize=(6, 6))
bars = ax.bar(
    ["Retrieval\n(FFN → target)", "Inhibition\n(Attn ← target)"],
    [retrieval_strength, inhibition_strength],
    color=["crimson", "dodgerblue"],
    edgecolor="black",
    linewidth=1.5,
)
ax.axhline(
    y=max(retrieval_strength, inhibition_strength),
    color="gray",
    linestyle=":",
    alpha=0.5,
)
ax.set_ylabel("Magnitude")
ax.set_title(
    f"Signal-to-Gate Ratio = {sgr:.2f}\n{'⚠ Negation Failure (SGR>1)' if sgr > 1 else '✓ Suppression (SGR<1)'}"
)
plt.tight_layout()
plt.savefig(f"{fig_dir}/05_sgr.png", dpi=150)
print(f"  Saved {fig_dir}/05_sgr.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — Summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SUMMARY — Single-Prompt Exploration")
print("=" * 72)
print(
    f"""
  Positive prompt : "{POSITIVE_PROMPT}"
  Negated prompt  : "{NEGATED_PROMPT}"
  Target token    : "{TARGET_TOKEN}"

  ┌─────────────────────────────────────────────────────────┐
  │  Behavioural Result                                     │
  │    Target rank (positive): {pos_info['target_rank']:<5}                        │
  │    Target rank (negated) : {neg_info['target_rank']:<5}                        │
  │    Negation failure      : {'YES' if negation_failure else 'NO':<5}                        │
  ├─────────────────────────────────────────────────────────┤
  │  DLA Aggregate                                          │
  │    FFN total  (pos): {ffn_pos_total:>8.3f}  (neg): {ffn_neg_total:>8.3f}       │
  │    Attn total (pos): {attn_pos_total:>8.3f}  (neg): {attn_neg_total:>8.3f}       │
  ├─────────────────────────────────────────────────────────┤
  │  Signal-to-Gate Ratio                                   │
  │    Retrieval  : {retrieval_strength:>8.3f}                              │
  │    Inhibition : {inhibition_strength:>8.3f}                              │
  │    SGR        : {sgr:>8.3f}  {'(FAILURE)' if sgr > 1 else '(SUCCESS)'}                     │
  ├─────────────────────────────────────────────────────────┤
  │  Crossover Point: {f'Layer {crossover}' if crossover is not None else 'NONE':>20}              │
  └─────────────────────────────────────────────────────────┘

  Figures saved to ./{fig_dir}/
"""
)

print("=" * 72)
print("DONE — Next steps:")
print("=" * 72)
print(
    """
  1. Scale up: Run this analysis over the full CounterFact dataset
     using src/dataset/load_dataset.py and src/benchmark/run_benchmark.py

  2. Per-head analysis: Decompose attention DLA per HEAD (not just per
     layer) to identify specific "Inhibition Heads" (à la Hanna et al.)

  3. Pythia comparison: Load pythia-160m and compare SGR distributions

  4. Artificial amplification: Scale inhibition head outputs by 2x, 3x
     to see if we can manually "fix" negation failures

  5. Statistical analysis: Correlate SGR with negation failure rate
     across hundreds of prompts
"""
)
