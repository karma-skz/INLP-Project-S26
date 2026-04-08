"""
hard_negation_audit.py
=======================
Hard Negation Audit — Semantic Pattern Analysis
-------------------------------------------------
Goal (from planned experiments):
  "Examine a broader set of hard negation examples to identify semantic
   patterns associated with hallucinations — specifically which entity types
   or prompt structures consistently produce negation failures."

Method
------
  1. Load the gpt2-small benchmark CSV (already computed DLA / SGR / failure).
  2. Re-join with the raw CounterFact HuggingFace dataset to recover rich
     metadata: relation_id, relation template, subject, target_true.
  3. Compute per-relation failure rates and rank them.
  4. Extract lightweight structural features from the prompt itself (prompt
     verb, subject length, target token length, pos_target_rank).
  5. Produce three publication-ready figures.

Outputs
-------
  figures/audit_relation_failure_rates.png  — per-relation failure bar chart
  figures/audit_prompt_structure.png         — structural feature heatmap
  figures/audit_sgr_by_relation.png          — SGR distribution per relation
  (console)                                  — ranked failure table

Usage
-----
  conda run -n inlp-project python hard_negation_audit.py
"""

from __future__ import annotations

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from datasets import load_dataset as hf_load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BENCHMARK_CSV = "results/gpt2-small_benchmark.csv"
FIG_DIR       = "figures"

# Wikidata relation ID → human-readable name (common CounterFact relations)
RELATION_LABELS: dict[str, str] = {
    "P103":  "mother tongue",
    "P106":  "occupation",
    "P17":   "country",
    "P19":   "place of birth",
    "P20":   "place of death",
    "P27":   "citizenship",
    "P276":  "location",
    "P30":   "continent",
    "P36":   "capital",
    "P37":   "official language",
    "P407":  "language of work",
    "P413":  "position played",
    "P449":  "original network",
    "P495":  "country of origin",
    "P530":  "diplomatic relation",
    "P740":  "location of formation",
    "P1001": "jurisdiction",
    "P1303": "instrument played",
    "P1376": "capital of",
    "P1412": "languages spoken",
    "P131":  "located in admin. div.",
    "P159":  "headquarters",
    "P178":  "developer",
    "P190":  "sister city",
    "P264":  "record label",
    "P279":  "subclass of",
    "P364":  "original language",
    "P39":   "position held",
    "P138":  "named after",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_benchmark(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df["negation_failure"].dtype == object:
        df["negation_failure"] = df["negation_failure"].map({"True": True, "False": False})
    df["sgr"] = pd.to_numeric(df["sgr"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def load_counterfact_metadata(max_id: int) -> pd.DataFrame:
    """Load the first max_id+1 entries of CounterFact and return as DataFrame."""
    print("Loading CounterFact metadata from HuggingFace …", flush=True)
    raw = hf_load_dataset("NeelNanda/counterfact-tracing", split="train")
    rows = []
    for i, entry in enumerate(raw):
        if i > max_id:
            break
        rows.append({
            "case_id":          i,
            "relation_id":      entry.get("relation_id", ""),
            "relation":         entry.get("relation", ""),
            "relation_prefix":  entry.get("relation_prefix", ""),
            "relation_suffix":  entry.get("relation_suffix", ""),
        })
    print(f"  → loaded {len(rows)} entries.")
    return pd.DataFrame(rows)


def extract_prompt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive lightweight structural features from the prompt string."""
    # Verb / relation suffix at the end of the positive prompt
    def extract_verb(prompt: str) -> str:
        p = prompt.strip().rstrip()
        # take last 1-3 words
        words = p.split()
        suffix = " ".join(words[-3:]) if len(words) >= 3 else p
        return suffix.lower()

    df = df.copy()
    df["prompt_verb"]        = df["positive_prompt"].apply(extract_verb)
    df["subject_word_count"] = df["subject"].apply(lambda s: len(str(s).split()))
    df["target_char_len"]    = df["target_token"].apply(lambda t: len(str(t).strip()))
    df["pos_logit_strength"] = df["pos_target_logit"]        # raw logit on positive
    df["logit_drop"]         = df["pos_target_logit"] - df["neg_target_logit"]  # how much "not" reduced logit
    return df


# ── Per-relation analysis ─────────────────────────────────────────────────────

def per_relation_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "relation_id" not in df.columns:
        return pd.DataFrame()

    rows = []
    for rel_id, grp in df.groupby("relation_id"):
        n          = len(grp)
        fails      = int(grp["negation_failure"].sum())
        fail_rate  = fails / n if n > 0 else float("nan")
        mean_sgr   = grp["sgr"].mean()
        med_logit  = grp["pos_target_logit"].median()
        mean_drop  = grp["logit_drop"].mean() if "logit_drop" in grp.columns else float("nan")
        label      = RELATION_LABELS.get(rel_id, rel_id)
        relation   = grp["relation"].iloc[0] if "relation" in grp.columns else rel_id
        rows.append({
            "relation_id":  rel_id,
            "relation":     relation,
            "label":        label,
            "n":            n,
            "failures":     fails,
            "fail_rate":    fail_rate,
            "mean_sgr":     mean_sgr,
            "med_pos_logit":med_logit,
            "mean_logit_drop": mean_drop,
        })
    return pd.DataFrame(rows).sort_values("fail_rate", ascending=False)


def print_relation_table(stats: pd.DataFrame) -> None:
    print("\n" + "─" * 85)
    print(f"  {'Relation ID':<12} {'Relation Type':<28} {'N':>4} {'Failures':>8} {'Fail %':>7} {'Mean SGR':>10}")
    print("─" * 85)
    for _, r in stats.iterrows():
        sgr_str = f"{r['mean_sgr']:.1f}" if not np.isnan(r["mean_sgr"]) else "  inf "
        print(f"  {r['relation_id']:<12} {r['label']:<28} {int(r['n']):>4} {int(r['failures']):>8} "
              f"{r['fail_rate']*100:>6.1f}% {sgr_str:>10}")
    print("─" * 85 + "\n")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_relation_failure_rates(stats: pd.DataFrame, fig_dir: str) -> None:
    """Horizontal bar chart of failure rate per relation, coloured by count."""
    if stats.empty:
        return

    # Only show relations with at least 2 samples
    plot_df = stats[stats["n"] >= 2].copy()
    if plot_df.empty:
        plot_df = stats.copy()

    # Sort ascending so highest failure at top when flipped
    plot_df = plot_df.sort_values("fail_rate")

    fig, ax = plt.subplots(figsize=(11, max(5, len(plot_df) * 0.5)))
    colors = plt.cm.RdYlGn_r(plot_df["fail_rate"].values)
    bars = ax.barh(plot_df["label"], plot_df["fail_rate"] * 100,
                   color=colors, edgecolor="white", linewidth=0.4)

    # Annotate with n
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"n={int(row['n'])}", va="center", ha="left", fontsize=8, color="#555")

    ax.set_xlabel("Negation Failure Rate (%)", fontsize=11)
    ax.set_title("Hard Negation Failure Rate by Relation Type\n(GPT-2 Small, CounterFact)", fontsize=12, fontweight="bold")
    ax.axvline(x=stats["fail_rate"].mean() * 100, color="navy", linestyle="--",
               linewidth=1.2, label=f"Mean={stats['fail_rate'].mean()*100:.1f}%")
    ax.legend(fontsize=9)
    ax.set_xlim(0, min(105, plot_df["fail_rate"].max() * 100 + 12))
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()

    path = os.path.join(fig_dir, "audit_relation_failure_rates.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_structural_heatmap(df: pd.DataFrame, fig_dir: str) -> None:
    """
    Heatmap: rows = failure / success, cols = structural features.
    Shows mean feature value for failed vs. succeeded samples.
    """
    feat_cols = [
        "pos_target_logit", "neg_target_logit", "logit_drop",
        "retrieval_strength", "inhibition_strength",
        "subject_word_count", "target_char_len",
        "pos_target_rank", "neg_target_rank",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]

    groups = {
        "Negation failure\n(hallucination)": df[df["negation_failure"]],
        "Negation success\n(correct)":        df[~df["negation_failure"]],
    }

    heat_data = []
    for grp_label, grp_df in groups.items():
        row = {"Group": grp_label}
        for col in feat_cols:
            row[col] = grp_df[col].mean()
        heat_data.append(row)

    heat_df = pd.DataFrame(heat_data).set_index("Group")[feat_cols]

    # Z-score each column for comparability
    heat_norm = (heat_df - heat_df.mean()) / (heat_df.std() + 1e-9)

    nice_names = {
        "pos_target_logit":   "Pos logit",
        "neg_target_logit":   "Neg logit",
        "logit_drop":         "Logit drop",
        "retrieval_strength": "Retrieval (FFN)",
        "inhibition_strength":"Inhibition (Attn)",
        "subject_word_count": "Subject words",
        "target_char_len":    "Target length",
        "pos_target_rank":    "Pos rank",
        "neg_target_rank":    "Neg rank",
    }
    heat_norm.columns = [nice_names.get(c, c) for c in heat_norm.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Structural Features: Failure vs. Success Samples", fontsize=12, fontweight="bold")

    # Left: heatmap
    sns.heatmap(
        heat_norm, ax=axes[0], annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.5, cbar_kws={"label": "Z-score"},
        annot_kws={"size": 9}
    )
    axes[0].set_title("Z-scored feature means (red = higher in group)", fontsize=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right", fontsize=9)

    # Right: violin of logit_drop by outcome
    ax2 = axes[1]
    fail_drop = df[df["negation_failure"]]["logit_drop"].dropna()
    succ_drop = df[~df["negation_failure"]]["logit_drop"].dropna()

    parts = ax2.violinplot([succ_drop, fail_drop], positions=[0, 1],
                           showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    parts["bodies"][0].set_facecolor("steelblue")
    parts["bodies"][1].set_facecolor("salmon")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Success", "Failure"], fontsize=10)
    ax2.set_ylabel('Logit drop\n(pos − neg logit)', fontsize=9)
    ax2.set_title("Logit drop distribution", fontsize=10)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_facecolor("#f8f9fa")

    plt.tight_layout()
    path = os.path.join(fig_dir, "audit_prompt_structure.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_sgr_by_relation(df: pd.DataFrame, stats: pd.DataFrame, fig_dir: str) -> None:
    """Box plot of SGR per relation (top-N most common relations)."""
    if stats.empty or "relation_id" not in df.columns:
        return

    # Take top 10 relations by sample count
    top_ids = stats.nlargest(10, "n")["relation_id"].tolist()
    plot_df = df[df["relation_id"].isin(top_ids)].copy()
    plot_df["relation_label"] = plot_df["relation_id"].map(
        dict(zip(stats["relation_id"], stats["label"]))
    )
    plot_df = plot_df.dropna(subset=["sgr"])

    if plot_df.empty:
        return

    order = (plot_df.groupby("relation_label")["negation_failure"]
             .mean().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(12, 5))
    fail_rates = dict(zip(stats["label"], stats["fail_rate"]))
    palette = {lbl: plt.cm.RdYlGn_r(fail_rates.get(lbl, 0)) for lbl in order}
    # Map colours as a list ordered by `order` for seaborn
    palette_list = [palette[lbl] for lbl in order]

    sns.boxplot(
        data=plot_df, x="relation_label", y="sgr", order=order,
        hue="relation_label", hue_order=order, palette=palette_list,
        legend=False, ax=ax, width=0.55, fliersize=3,
        flierprops={"alpha": 0.4}
    )
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2, label="SGR = 1")
    ax.set_yscale("log")
    ax.set_xlabel("Relation Type", fontsize=11)
    ax.set_ylabel("SGR (log scale)", fontsize=11)
    ax.set_title("SGR Distribution by Relation Type\n(coloured by failure rate — red = high failure)",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")

    # Annotate failure % on each box
    for i, lbl in enumerate(order):
        fr = fail_rates.get(lbl, float("nan"))
        n  = stats[stats["label"] == lbl]["n"].values
        n_str = f"n={int(n[0])}" if len(n) else ""
        ax.text(i, ax.get_ylim()[0] * 1.05, f"{fr*100:.0f}%\n{n_str}",
                ha="center", va="bottom", fontsize=7.5, color="#333")

    plt.tight_layout()
    path = os.path.join(fig_dir, "audit_sgr_by_relation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Qualitative examples ──────────────────────────────────────────────────────

def print_failure_examples(df: pd.DataFrame, n: int = 10) -> None:
    fails = df[df["negation_failure"]].copy()
    if fails.empty:
        return
    fails = fails.sort_values("sgr", ascending=False).head(n)
    print(f"\n{'─'*90}")
    print(f"  Top {n} negation failures (highest SGR first)")
    print(f"{'─'*90}")
    for _, row in fails.iterrows():
        rel = row.get("relation_id", "?") + " — " + row.get("label", "?") \
              if "label" in row else row.get("relation_id", "?")
        sgr_str = f"{row['sgr']:.1f}" if not np.isnan(row.get("sgr", float("nan"))) else "inf"
        print(f"\n  [{rel}]")
        print(f"  Prompt  : {row['positive_prompt']}")
        print(f"  Target  : '{row['target_token'].strip()}'  |  SGR={sgr_str}"
              f"  |  pos_rank={int(row['pos_target_rank'])}  neg_rank={int(row['neg_target_rank'])}")
    print(f"{'─'*90}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Hard Negation Audit — Semantic Pattern Analysis")
    print("=" * 65)

    if not os.path.exists(BENCHMARK_CSV):
        print(f"ERROR: benchmark CSV not found at {BENCHMARK_CSV}")
        return

    # 1. Load benchmark results
    df = load_benchmark(BENCHMARK_CSV)
    print(f"Loaded benchmark: {len(df)} samples, {df['negation_failure'].sum()} failures "
          f"({df['negation_failure'].mean()*100:.1f}%)")

    # 2. Join CounterFact metadata by case_id / row-index
    max_id = int(df["case_id"].max())
    meta   = load_counterfact_metadata(max_id)
    df     = df.merge(meta, on="case_id", how="left")

    # 3. Extract prompt-structural features
    df = extract_prompt_features(df)

    # 4. Per-relation statistics
    stats = per_relation_stats(df)
    if not stats.empty:
        print_relation_table(stats)

    # 5. Qualitative failure examples (with relation labels merged)
    if "relation_id" in df.columns and not stats.empty:
        label_map = dict(zip(stats["relation_id"], stats["label"]))
        df["label"] = df["relation_id"].map(label_map)
    print_failure_examples(df, n=10)

    # 6. Summary statistics
    total_fail = int(df["negation_failure"].sum())
    total      = len(df)
    print(f"\n  Overall failure rate : {total_fail}/{total} = {total_fail/total*100:.1f}%")

    if not stats.empty:
        worst = stats.iloc[0]
        best  = stats[stats["n"] >= 2].iloc[-1]
        print(f"  Worst relation  : '{worst['label']}' ({worst['relation_id']}) — "
              f"{worst['fail_rate']*100:.1f}% failure rate (n={int(worst['n'])})")
        print(f"  Best relation   : '{best['label']}' ({best['relation_id']}) — "
              f"{best['fail_rate']*100:.1f}% failure rate (n={int(best['n'])})")

    fail_high = df[df["negation_failure"] & (df["subject_word_count"] >= 2)]
    fail_low  = df[df["negation_failure"] & (df["subject_word_count"] < 2)]
    print(f"\n  Failures by subject length:")
    print(f"    Short subject (1 word)  : {len(fail_low)}")
    print(f"    Long subject (≥2 words) : {len(fail_high)}")

    # Logit drop comparison
    drop_fail = df[df["negation_failure"]]["logit_drop"].mean()
    drop_succ = df[~df["negation_failure"]]["logit_drop"].mean()
    print(f"\n  Mean logit drop (pos_logit − neg_logit):")
    print(f"    Failure cases  : {drop_fail:.3f}")
    print(f"    Success cases  : {drop_succ:.3f}")
    print(f"  → Failures see {drop_fail/drop_succ:.2f}× smaller logit reduction from 'not'")

    # 7. Figures
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_relation_failure_rates(stats, FIG_DIR)
    plot_structural_heatmap(df, FIG_DIR)
    plot_sgr_by_relation(df, stats, FIG_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
