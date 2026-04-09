from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset as hf_load_dataset

from src.utils import load_benchmark_dataframe, resolve_benchmark_csv


matplotlib.use("Agg")


RELATION_LABELS: dict[str, str] = {
    "P103": "mother tongue",
    "P106": "occupation",
    "P17": "country",
    "P19": "place of birth",
    "P20": "place of death",
    "P27": "citizenship",
    "P276": "location",
    "P30": "continent",
    "P36": "capital",
    "P37": "official language",
    "P39": "position held",
    "P131": "located in admin. div.",
    "P138": "named after",
    "P159": "headquarters",
    "P178": "developer",
    "P190": "sister city",
    "P264": "record label",
    "P279": "subclass of",
    "P364": "original language",
    "P407": "language of work",
    "P413": "position played",
    "P449": "original network",
    "P495": "country of origin",
    "P530": "diplomatic relation",
    "P740": "location of formation",
    "P1001": "jurisdiction",
    "P1303": "instrument played",
    "P1376": "capital of",
    "P1412": "languages spoken",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the semantic audit on a benchmark CSV")
    parser.add_argument("--results_dir", default="results", help="Directory containing benchmark CSVs")
    parser.add_argument("--model", default="gpt2-small", help="Model benchmark to inspect")
    parser.add_argument("--negator_suffix", default=" not", help="Negator suffix used for the benchmark CSV")
    parser.add_argument("--results_csv", help="Explicit benchmark CSV path. Overrides --results_dir/--model/--negator_suffix")
    parser.add_argument("--fig_dir", default="figures", help="Directory to save figures into")
    return parser.parse_args()


def load_counterfact_metadata(max_id: int) -> pd.DataFrame:
    print("Loading CounterFact metadata from HuggingFace ...", flush=True)
    raw = hf_load_dataset("NeelNanda/counterfact-tracing", split="train")
    rows = []
    for index, entry in enumerate(raw):
        if index > max_id:
            break
        rows.append(
            {
                "case_id": index,
                "relation_id": entry.get("relation_id", ""),
                "relation": entry.get("relation", ""),
                "relation_prefix": entry.get("relation_prefix", ""),
                "relation_suffix": entry.get("relation_suffix", ""),
            }
        )
    print(f"  -> loaded {len(rows)} entries.")
    return pd.DataFrame(rows)


def extract_prompt_features(df: pd.DataFrame) -> pd.DataFrame:
    def extract_verb(prompt: str) -> str:
        words = str(prompt).strip().split()
        return " ".join(words[-3:]).lower() if len(words) >= 3 else str(prompt).lower()

    enriched = df.copy()
    enriched["prompt_verb"] = enriched["positive_prompt"].apply(extract_verb)
    enriched["subject_word_count"] = enriched["subject"].apply(lambda value: len(str(value).split()))
    enriched["target_char_len"] = enriched["target_token"].apply(lambda value: len(str(value).strip()))
    enriched["pos_logit_strength"] = enriched["pos_target_logit"]
    enriched["logit_drop"] = enriched["pos_target_logit"] - enriched["neg_target_logit"]
    return enriched


def per_relation_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "relation_id" not in df.columns:
        return pd.DataFrame()

    rows = []
    for relation_id, group in df.groupby("relation_id"):
        label = RELATION_LABELS.get(relation_id, relation_id)
        rows.append(
            {
                "relation_id": relation_id,
                "relation": group["relation"].iloc[0] if "relation" in group.columns else relation_id,
                "label": label,
                "n": len(group),
                "failures": int(group["negation_failure"].sum()),
                "fail_rate": float(group["negation_failure"].mean()),
                "mean_sgr": float(group["sgr"].mean()),
                "med_pos_logit": float(group["pos_target_logit"].median()),
                "mean_logit_drop": float(group["logit_drop"].mean()) if "logit_drop" in group.columns else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values("fail_rate", ascending=False)


def print_relation_table(stats: pd.DataFrame) -> None:
    print("\n" + "-" * 85)
    print(f"  {'Relation ID':<12} {'Relation Type':<28} {'N':>4} {'Failures':>8} {'Fail %':>7} {'Mean SGR':>10}")
    print("-" * 85)
    for _, row in stats.iterrows():
        sgr_string = f"{row['mean_sgr']:.1f}" if not np.isnan(row["mean_sgr"]) else "inf"
        print(
            f"  {row['relation_id']:<12} {row['label']:<28} {int(row['n']):>4} {int(row['failures']):>8} "
            f"{row['fail_rate'] * 100:>6.1f}% {sgr_string:>10}"
        )
    print("-" * 85 + "\n")


def plot_relation_failure_rates(stats: pd.DataFrame, fig_dir: str) -> str | None:
    if stats.empty:
        return None

    plot_df = stats[stats["n"] >= 2].copy()
    if plot_df.empty:
        plot_df = stats.copy()
    plot_df = plot_df.sort_values("fail_rate")

    fig, ax = plt.subplots(figsize=(11, max(5, len(plot_df) * 0.5)))
    colors = plt.cm.RdYlGn_r(plot_df["fail_rate"].values)
    bars = ax.barh(plot_df["label"], plot_df["fail_rate"] * 100, color=colors, edgecolor="white", linewidth=0.4)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"n={int(row['n'])}", va="center", ha="left", fontsize=8, color="#555")

    ax.set_xlabel("Negation Failure Rate (%)", fontsize=11)
    ax.set_title("Hard Negation Failure Rate by Relation Type", fontsize=12, fontweight="bold")
    ax.axvline(x=stats["fail_rate"].mean() * 100, color="navy", linestyle="--", linewidth=1.2, label=f"Mean={stats['fail_rate'].mean() * 100:.1f}%")
    ax.legend(fontsize=9)
    ax.set_xlim(0, min(105, plot_df["fail_rate"].max() * 100 + 12))
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()

    path = os.path.join(fig_dir, "audit_relation_failure_rates.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_structural_heatmap(df: pd.DataFrame, fig_dir: str) -> str:
    feature_columns = [
        "pos_target_logit",
        "neg_target_logit",
        "logit_drop",
        "retrieval_strength",
        "inhibition_strength",
        "subject_word_count",
        "target_char_len",
        "pos_target_rank",
        "neg_target_rank",
    ]
    feature_columns = [column for column in feature_columns if column in df.columns]

    groups = {
        "Negation failure\n(hallucination)": df[df["negation_failure"]],
        "Negation success\n(correct)": df[~df["negation_failure"]],
    }

    heat_rows = []
    for group_name, group_df in groups.items():
        row = {"Group": group_name}
        for column in feature_columns:
            row[column] = group_df[column].mean()
        heat_rows.append(row)

    heat_df = pd.DataFrame(heat_rows).set_index("Group")[feature_columns]
    heat_norm = (heat_df - heat_df.mean()) / (heat_df.std() + 1e-9)

    pretty_names = {
        "pos_target_logit": "Pos logit",
        "neg_target_logit": "Neg logit",
        "logit_drop": "Logit drop",
        "retrieval_strength": "Retrieval (FFN)",
        "inhibition_strength": "Inhibition (Attn)",
        "subject_word_count": "Subject words",
        "target_char_len": "Target length",
        "pos_target_rank": "Pos rank",
        "neg_target_rank": "Neg rank",
    }
    heat_norm.columns = [pretty_names.get(column, column) for column in heat_norm.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Structural Features: Failure vs. Success Samples", fontsize=12, fontweight="bold")

    sns.heatmap(
        heat_norm,
        ax=axes[0],
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Z-score"},
        annot_kws={"size": 9},
    )
    axes[0].set_title("Z-scored feature means", fontsize=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right", fontsize=9)

    ax2 = axes[1]
    success_drop = df[~df["negation_failure"]]["logit_drop"].dropna()
    failure_drop = df[df["negation_failure"]]["logit_drop"].dropna()

    parts = ax2.violinplot([success_drop, failure_drop], positions=[0, 1], showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
    parts["bodies"][0].set_facecolor("steelblue")
    parts["bodies"][1].set_facecolor("salmon")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Success", "Failure"], fontsize=10)
    ax2.set_ylabel("Logit drop\n(pos - neg logit)", fontsize=9)
    ax2.set_title("Logit drop distribution", fontsize=10)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_facecolor("#f8f9fa")

    plt.tight_layout()
    path = os.path.join(fig_dir, "audit_prompt_structure.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_sgr_by_relation(df: pd.DataFrame, stats: pd.DataFrame, fig_dir: str) -> str | None:
    if stats.empty or "relation_id" not in df.columns:
        return None

    top_ids = stats.nlargest(10, "n")["relation_id"].tolist()
    plot_df = df[df["relation_id"].isin(top_ids)].copy()
    plot_df["relation_label"] = plot_df["relation_id"].map(dict(zip(stats["relation_id"], stats["label"])))
    plot_df = plot_df.dropna(subset=["sgr"])

    if plot_df.empty:
        return None

    order = plot_df.groupby("relation_label")["negation_failure"].mean().sort_values(ascending=False).index.tolist()
    fail_rates = dict(zip(stats["label"], stats["fail_rate"]))
    palette = [plt.cm.RdYlGn_r(fail_rates.get(label, 0.0)) for label in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=plot_df,
        x="relation_label",
        y="sgr",
        order=order,
        hue="relation_label",
        hue_order=order,
        palette=palette,
        legend=False,
        ax=ax,
        width=0.55,
        fliersize=3,
        flierprops={"alpha": 0.4},
    )
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2, label="SGR = 1")
    ax.set_yscale("log")
    ax.set_xlabel("Relation Type", fontsize=11)
    ax.set_ylabel("SGR (log scale)", fontsize=11)
    ax.set_title("SGR Distribution by Relation Type", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f8f9fa")

    for idx, label in enumerate(order):
        fail_rate = fail_rates.get(label, float("nan"))
        counts = stats[stats["label"] == label]["n"].values
        count_label = f"n={int(counts[0])}" if len(counts) else ""
        ax.text(idx, ax.get_ylim()[0] * 1.05, f"{fail_rate * 100:.0f}%\n{count_label}", ha="center", va="bottom", fontsize=7.5, color="#333")

    plt.tight_layout()
    path = os.path.join(fig_dir, "audit_sgr_by_relation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def print_failure_examples(df: pd.DataFrame, count: int = 10) -> None:
    failures = df[df["negation_failure"]].copy()
    if failures.empty:
        return

    failures = failures.sort_values("sgr", ascending=False).head(count)
    print(f"\n{'-' * 90}")
    print(f"  Top {count} negation failures (highest SGR first)")
    print(f"{'-' * 90}")
    for _, row in failures.iterrows():
        relation = f"{row.get('relation_id', '?')} — {row.get('label', '?')}" if "label" in row else row.get("relation_id", "?")
        sgr_string = f"{row['sgr']:.1f}" if not np.isnan(row.get("sgr", float("nan"))) else "inf"
        print(f"\n  [{relation}]")
        print(f"  Prompt  : {row['positive_prompt']}")
        print(
            f"  Target  : '{row['target_token'].strip()}'  |  SGR={sgr_string}"
            f"  |  pos_rank={int(row['pos_target_rank'])}  neg_rank={int(row['neg_target_rank'])}"
        )
    print(f"{'-' * 90}\n")


def run_semantic_audit(
    results_csv: str | None = None,
    results_dir: str = "results",
    model_name: str = "gpt2-small",
    negator_suffix: str = " not",
    fig_dir: str = "figures",
) -> list[str]:
    print("=" * 65)
    print("  Hard Negation Audit — Semantic Pattern Analysis")
    print("=" * 65)

    csv_path = results_csv or str(resolve_benchmark_csv(results_dir, model_name, negator_suffix))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark CSV not found at {csv_path}")

    df = load_benchmark_dataframe(csv_path)
    print(f"Loaded benchmark: {len(df)} samples, {df['negation_failure'].sum()} failures ({df['negation_failure'].mean() * 100:.1f}%)")

    max_id = int(df["case_id"].max())
    metadata = load_counterfact_metadata(max_id)
    df = df.merge(metadata, on="case_id", how="left")
    df = extract_prompt_features(df)

    stats = per_relation_stats(df)
    if not stats.empty:
        print_relation_table(stats)

    if "relation_id" in df.columns and not stats.empty:
        label_map = dict(zip(stats["relation_id"], stats["label"]))
        df["label"] = df["relation_id"].map(label_map)
    print_failure_examples(df, count=10)

    total_failures = int(df["negation_failure"].sum())
    total = len(df)
    print(f"\n  Overall failure rate : {total_failures}/{total} = {total_failures / total * 100:.1f}%")

    if not stats.empty:
        worst = stats.iloc[0]
        filtered = stats[stats["n"] >= 2]
        best = filtered.iloc[-1] if not filtered.empty else stats.iloc[-1]
        print(f"  Worst relation  : '{worst['label']}' ({worst['relation_id']}) — {worst['fail_rate'] * 100:.1f}% failure rate (n={int(worst['n'])})")
        print(f"  Best relation   : '{best['label']}' ({best['relation_id']}) — {best['fail_rate'] * 100:.1f}% failure rate (n={int(best['n'])})")

    os.makedirs(fig_dir, exist_ok=True)
    paths = [
        path
        for path in [
            plot_relation_failure_rates(stats, fig_dir),
            plot_structural_heatmap(df, fig_dir),
            plot_sgr_by_relation(df, stats, fig_dir),
        ]
        if path is not None
    ]

    print("\nDone.")
    return paths


def main():
    args = parse_args()
    try:
        run_semantic_audit(
            results_csv=args.results_csv,
            results_dir=args.results_dir,
            model_name=args.model,
            negator_suffix=args.negator_suffix,
            fig_dir=args.fig_dir,
        )
    except FileNotFoundError as exc:
        print(exc)


if __name__ == "__main__":
    main()
