"""
Generate a detailed qualitative analysis package from cross-model benchmark outputs.

Outputs are written under reports/final:
- qualitative.md
- qualitative_selected_case_ids.csv
- qualitative_selected_case_metrics.csv
- qualitative_case_comparison.png
- qualitative_layer_shift_heatmap.png
- qualitative_cases/case_<id>/...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import benchmark_csv_path, load_benchmark_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Detailed qualitative analysis from cross-model benchmark CSVs")
    parser.add_argument("--models", nargs=2, default=["gpt2-small", "pythia-160m"], help="Exactly two model names")
    parser.add_argument("--results_dir", default="results/cross_model", help="Directory containing benchmark CSVs")
    parser.add_argument("--negator_suffix", default=" not", help="Negator suffix used in benchmark CSV names")
    parser.add_argument("--n_cases", type=int, default=10, help="Number of shared cases to analyse")
    parser.add_argument("--figures_output_dir", default="figures/qualitative", help="Output directory for graphs/plots")
    parser.add_argument("--reports_output_dir", default="reports", help="Output directory for markdown reports")
    parser.add_argument("--results_output_dir", default="results/qualitative", help="Output directory for CSV/data artifacts")
    return parser.parse_args()


def _parse_layer_str(value: object | None) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [float(x) for x in text.split("|")]


def _safe_float(x) -> float:
    if pd.isna(x):
        return float("nan")
    return float(x)


def _load_model_df(model_name: str, results_dir: str, negator_suffix: str) -> pd.DataFrame:
    csv_path = benchmark_csv_path(results_dir, model_name, negator_suffix)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing benchmark CSV for {model_name}: {csv_path}")
    df = load_benchmark_dataframe(csv_path)
    if "case_id" not in df.columns:
        raise ValueError(f"Missing case_id column in {csv_path}")
    return df


def _mismatch_rate_row(row: pd.Series) -> bool:
    # mismatch when boolean negation-failure criterion disagrees with SGR (>1 implies failure)
    fail = bool(row["negation_failure"])
    sgr = _safe_float(row["sgr"])
    if np.isnan(sgr):
        return False
    return ((not fail) and (sgr > 1.0)) or (fail and (sgr <= 1.0))


def _select_case_ids(df_a: pd.DataFrame, df_b: pd.DataFrame, n_cases: int) -> list[int]:
    common_ids = sorted(set(df_a["case_id"]).intersection(set(df_b["case_id"])))
    if not common_ids:
        raise ValueError("No shared case IDs between models.")

    a = df_a[df_a["case_id"].isin(common_ids)].copy().set_index("case_id")
    b = df_b[df_b["case_id"].isin(common_ids)].copy().set_index("case_id")

    rows = []
    for case_id in common_ids:
        ra = a.loc[case_id]
        rb = b.loc[case_id]
        a_drop = float(ra["neg_target_rank"] - ra["pos_target_rank"])
        b_drop = float(rb["neg_target_rank"] - rb["pos_target_rank"])
        a_fail = bool(ra["negation_failure"])
        b_fail = bool(rb["negation_failure"])
        a_mis = _mismatch_rate_row(ra)
        b_mis = _mismatch_rate_row(rb)
        rows.append(
            {
                "case_id": int(case_id),
                "avg_rank_drop": (a_drop + b_drop) / 2.0,
                "max_rank_drop": max(a_drop, b_drop),
                "any_failure": a_fail or b_fail,
                "both_failure": a_fail and b_fail,
                "failure_disagreement": a_fail != b_fail,
                "any_mismatch": a_mis or b_mis,
                "mismatch_disagreement": a_mis != b_mis,
                "rank_drop_gap": abs(a_drop - b_drop),
            }
        )

    sel_df = pd.DataFrame(rows).sort_values("avg_rank_drop", ascending=False)

    chosen: list[int] = []
    seen: set[int] = set()

    def add_from(bucket: pd.DataFrame, k: int):
        for _, r in bucket.iterrows():
            cid = int(r["case_id"])
            if cid in seen:
                continue
            chosen.append(cid)
            seen.add(cid)
            if len(chosen) >= k:
                break

    # 1) disagreement-heavy cases
    add_from(
        sel_df[sel_df["failure_disagreement"] | sel_df["mismatch_disagreement"]]
        .sort_values(["rank_drop_gap", "avg_rank_drop"], ascending=[False, False]),
        min(3, n_cases),
    )

    # 2) strongest failures
    add_from(
        sel_df[sel_df["any_failure"]].sort_values(["both_failure", "max_rank_drop"], ascending=[False, False]),
        min(6, n_cases),
    )

    # 3) strongest non-failure but high-rank-shift cases
    add_from(
        sel_df[~sel_df["any_failure"]].sort_values("avg_rank_drop", ascending=False),
        n_cases,
    )

    # fill remainder
    if len(chosen) < n_cases:
        add_from(sel_df.sort_values("avg_rank_drop", ascending=False), n_cases)

    return chosen[:n_cases]


def _case_layer_dataframe(row: pd.Series, model_name: str) -> pd.DataFrame:
    ffn_pos = _parse_layer_str(row.get("ffn_dla_pos_str"))
    ffn_neg = _parse_layer_str(row.get("ffn_dla_neg_str"))
    attn_pos = _parse_layer_str(row.get("attn_dla_pos_str"))
    attn_neg = _parse_layer_str(row.get("attn_dla_neg_str"))

    n_layers = min(len(ffn_pos), len(ffn_neg), len(attn_pos), len(attn_neg))
    data = []
    for layer in range(n_layers):
        pos_total = ffn_pos[layer] + attn_pos[layer]
        neg_total = ffn_neg[layer] + attn_neg[layer]
        data.append(
            {
                "model_name": model_name,
                "layer": layer,
                "ffn_pos": ffn_pos[layer],
                "ffn_neg": ffn_neg[layer],
                "attn_pos": attn_pos[layer],
                "attn_neg": attn_neg[layer],
                "ffn_delta": ffn_neg[layer] - ffn_pos[layer],
                "attn_delta": attn_neg[layer] - attn_pos[layer],
                "pos_total": pos_total,
                "neg_total": neg_total,
                "total_delta": neg_total - pos_total,
            }
        )
    return pd.DataFrame(data)


def _plot_case_layer_lines(case_layer_df: pd.DataFrame, output_path: Path, case_id: int):
    models = list(case_layer_df["model_name"].unique())
    fig, axes = plt.subplots(len(models), 2, figsize=(14, 4.2 * len(models)), squeeze=False)

    for i, model_name in enumerate(models):
        mdf = case_layer_df[case_layer_df["model_name"] == model_name]
        layers = mdf["layer"].to_numpy()

        ax_left = axes[i, 0]
        ax_left.plot(layers, mdf["ffn_pos"], marker="o", label="FFN pos")
        ax_left.plot(layers, mdf["ffn_neg"], marker="o", label="FFN neg")
        ax_left.plot(layers, mdf["attn_pos"], marker="s", label="Attn pos")
        ax_left.plot(layers, mdf["attn_neg"], marker="s", label="Attn neg")
        ax_left.set_title(f"{model_name}: per-layer components")
        ax_left.set_xlabel("Layer")
        ax_left.set_ylabel("DLA contribution")
        ax_left.grid(alpha=0.3)
        ax_left.legend(fontsize=8)

        ax_right = axes[i, 1]
        width = 0.35
        ax_right.bar(layers - width / 2, mdf["ffn_delta"], width=width, label="FFN Δ (neg-pos)")
        ax_right.bar(layers + width / 2, mdf["attn_delta"], width=width, label="Attn Δ (neg-pos)")
        ax_right.axhline(0.0, color="black", linewidth=1)
        ax_right.set_title(f"{model_name}: negation-induced shift")
        ax_right.set_xlabel("Layer")
        ax_right.set_ylabel("Δ contribution")
        ax_right.grid(alpha=0.3)
        ax_right.legend(fontsize=8)

    fig.suptitle(f"Case {case_id}: layer-wise behavior", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_case_behavior(case_metrics_df: pd.DataFrame, output_path: Path, case_id: int):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    models = case_metrics_df["model_name"].tolist()
    x = np.arange(len(models))
    width = 0.24

    # logits
    axes[0].bar(x - width, case_metrics_df["pos_target_logit"], width=width, label="positive")
    axes[0].bar(x, case_metrics_df["neg_target_logit"], width=width, label="negated")
    axes[0].bar(x + width, case_metrics_df["patched_best_logit"], width=width, label="best-patched")
    axes[0].set_title("Target logit")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20)

    # probs
    axes[1].bar(x - width, case_metrics_df["pos_target_prob"], width=width, label="positive")
    axes[1].bar(x, case_metrics_df["neg_target_prob"], width=width, label="negated")
    axes[1].bar(x + width, case_metrics_df["patched_best_prob"], width=width, label="best-patched")
    axes[1].set_title("Target probability")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20)

    # ranks
    axes[2].bar(x - width, case_metrics_df["pos_target_rank"], width=width, label="positive")
    axes[2].bar(x, case_metrics_df["neg_target_rank"], width=width, label="negated")
    axes[2].bar(x + width, case_metrics_df["patched_best_rank"], width=width, label="best-patched")
    axes[2].set_title("Target rank (lower is better)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Case {case_id}: behavioral metrics", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _layer_concentration_stats(case_layer_df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for model_name, mdf in case_layer_df.groupby("model_name"):
        if mdf.empty:
            continue
        abs_delta = mdf["total_delta"].abs()
        total_abs = float(abs_delta.sum())
        if total_abs <= 0:
            concentration = 0.0
            top3_share = 0.0
        else:
            concentration = float(abs_delta.max() / total_abs)
            top3_share = float(abs_delta.nlargest(3).sum() / total_abs)

        dominant_idx = int(abs_delta.idxmax())
        dominant_layer = int(pd.to_numeric(mdf.loc[dominant_idx, "layer"], errors="coerce"))
        dominant_shift_value = float(pd.to_numeric(mdf.loc[dominant_idx, "total_delta"], errors="coerce"))

        stats.append(
            {
                "model_name": model_name,
                "dominant_shift_layer": dominant_layer,
                "dominant_shift_value": dominant_shift_value,
                "total_abs_shift": total_abs,
                "top3_layer_shift_share": top3_share,
                "single_layer_shift_share": concentration,
            }
        )
    return pd.DataFrame(stats)


def _make_global_figures(selected_metrics_df: pd.DataFrame, case_layer_all_df: pd.DataFrame, output_dir: Path):
    # per-case average rank drop across models
    case_rank = (
        selected_metrics_df.assign(rank_drop=lambda d: d["neg_target_rank"] - d["pos_target_rank"])
        .groupby("case_id", as_index=False)["rank_drop"]
        .mean()
        .sort_values("rank_drop", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(case_rank["case_id"].astype(str), case_rank["rank_drop"], color="#4C78A8")
    ax.set_title("Average rank drop (negated - positive) across models")
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Rank drop")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "qualitative_case_comparison.png", dpi=180)
    plt.close(fig)

    # heatmap: total_delta by layer and case, separated by model (saved as two panels)
    models = list(selected_metrics_df["model_name"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8), squeeze=False)
    for i, model_name in enumerate(models):
        mdf = case_layer_all_df[case_layer_all_df["model_name"] == model_name]
        pivot = mdf.pivot_table(index="case_id", columns="layer", values="total_delta", aggfunc="mean")
        pivot = pivot.sort_index()
        im = axes[0, i].imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
        axes[0, i].set_title(f"{model_name}: total Δ by layer")
        axes[0, i].set_xlabel("Layer")
        axes[0, i].set_ylabel("Case ID")
        axes[0, i].set_xticks(np.arange(pivot.shape[1]))
        axes[0, i].set_yticks(np.arange(pivot.shape[0]))
        axes[0, i].set_yticklabels(pivot.index.astype(int).tolist())
        fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_dir / "qualitative_layer_shift_heatmap.png", dpi=180)
    plt.close(fig)


def _build_markdown_report(
    output_path: Path,
    figures_output_dir: Path,
    results_output_dir: Path,
    selected_case_ids: list[int],
    selected_metrics_df: pd.DataFrame,
    per_case_layer_stats: dict[int, pd.DataFrame],
):
    report_dir = output_path.parent

    def _rel(target: Path) -> str:
        return os.path.relpath(target, start=report_dir).replace("\\", "/")

    lines: list[str] = []
    lines.append("# Detailed qualitative analysis from cross-model outputs")
    lines.append("")
    lines.append("This report uses the benchmark outputs in results/cross_model and compares the same 10 case IDs across both models.")
    lines.append("")
    lines.append("## Selected shared case IDs")
    lines.append("")
    lines.append(", ".join(str(cid) for cid in selected_case_ids))
    lines.append("")
    lines.append("## Global artifacts")
    lines.append("")
    selected_ids_csv = results_output_dir / "qualitative_selected_case_ids.csv"
    selected_metrics_csv = results_output_dir / "qualitative_selected_case_metrics.csv"
    global_rank_plot = figures_output_dir / "qualitative_case_comparison.png"
    global_heatmap_plot = figures_output_dir / "qualitative_layer_shift_heatmap.png"
    lines.append(f"- [{_rel(selected_ids_csv)}]({_rel(selected_ids_csv)})")
    lines.append(f"- [{_rel(selected_metrics_csv)}]({_rel(selected_metrics_csv)})")
    lines.append(f"- [{_rel(global_rank_plot)}]({_rel(global_rank_plot)})")
    lines.append(f"- [{_rel(global_heatmap_plot)}]({_rel(global_heatmap_plot)})")
    lines.append("")

    lines.append("## Case-by-case analysis")
    lines.append("")

    for case_id in selected_case_ids:
        cdf = selected_metrics_df[selected_metrics_df["case_id"] == case_id].sort_values("model_name")
        if cdf.empty:
            continue

        prompt = str(cdf.iloc[0]["positive_prompt"])
        neg_prompt = str(cdf.iloc[0]["negated_prompt"])
        target = str(cdf.iloc[0]["target_token"])

        lines.append(f"### Case {case_id}")
        lines.append("")
        lines.append(f"- Subject: {cdf.iloc[0]['subject']}")
        lines.append(f"- Positive prompt: {prompt}")
        lines.append(f"- Negated prompt: {neg_prompt}")
        lines.append(f"- Target token: {target!r}")
        lines.append("")

        case_results_dir = results_output_dir / "qualitative_cases" / f"case_{case_id}"
        case_figures_dir = figures_output_dir / "qualitative_cases" / f"case_{case_id}"
        case_metrics_csv = _rel(case_results_dir / "case_metrics.csv")
        case_layer_csv = _rel(case_results_dir / "layer_metrics.csv")
        case_layer_concentration_csv = _rel(case_results_dir / "layer_concentration.csv")
        case_layer_plot = _rel(case_figures_dir / "layer_behavior.png")
        case_behavior_plot = _rel(case_figures_dir / "behavior_summary.png")
        lines.append("Artifacts:")
        lines.append("")
        lines.append(f"- [{case_metrics_csv}]({case_metrics_csv})")
        lines.append(f"- [{case_layer_csv}]({case_layer_csv})")
        lines.append(f"- [{case_layer_concentration_csv}]({case_layer_concentration_csv})")
        lines.append(f"- [{case_layer_plot}]({case_layer_plot})")
        lines.append(f"- [{case_behavior_plot}]({case_behavior_plot})")
        lines.append("")

        lines.append("Model-wise behavioral notes:")
        lines.append("")
        for _, row in cdf.iterrows():
            rank_drop = int(row["neg_target_rank"] - row["pos_target_rank"])
            patch_recovery = int(row["neg_target_rank"] - row["patched_best_rank"])
            lines.append(
                f"- {row['model_name']}: failure={bool(row['negation_failure'])}, "
                f"SGR={float(row['sgr']):.3f}, rank drop={rank_drop}, best patch={row['best_patch_type']}@L{int(row['best_patch_layer'])}, "
                f"patch rank recovery={patch_recovery}."
            )

        layer_stats = per_case_layer_stats.get(case_id)
        if layer_stats is not None and not layer_stats.empty:
            lines.append("")
            lines.append("Layer-shift concentration notes:")
            lines.append("")
            for _, ls in layer_stats.iterrows():
                lines.append(
                    f"- {ls['model_name']}: dominant shift at layer {int(ls['dominant_shift_layer'])} "
                    f"(Δ={float(ls['dominant_shift_value']):.3f}), top-3 layers explain "
                    f"{100.0 * float(ls['top3_layer_shift_share']):.1f}% of absolute shift."
                )

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if len(args.models) != 2:
        raise ValueError("Please provide exactly two models for paired qualitative analysis.")

    figures_output_dir = Path(args.figures_output_dir)
    reports_output_dir = Path(args.reports_output_dir)
    results_output_dir = Path(args.results_output_dir)
    figures_output_dir.mkdir(parents=True, exist_ok=True)
    reports_output_dir.mkdir(parents=True, exist_ok=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)

    model_a, model_b = args.models
    df_a = _load_model_df(model_a, args.results_dir, args.negator_suffix)
    df_b = _load_model_df(model_b, args.results_dir, args.negator_suffix)

    selected_case_ids = _select_case_ids(df_a, df_b, args.n_cases)

    selected_rows = []
    case_layer_frames = []
    per_case_layer_stats: dict[int, pd.DataFrame] = {}

    for case_id in selected_case_ids:
        case_results_dir = results_output_dir / "qualitative_cases" / f"case_{case_id}"
        case_figures_dir = figures_output_dir / "qualitative_cases" / f"case_{case_id}"
        case_results_dir.mkdir(parents=True, exist_ok=True)
        case_figures_dir.mkdir(parents=True, exist_ok=True)

        case_metrics_rows = []
        case_layer_rows = []

        for model_name, df in [(model_a, df_a), (model_b, df_b)]:
            row = df[df["case_id"] == case_id].iloc[0]

            # Compute best patch directly from layer deltas available in benchmark outputs
            layer_df = _case_layer_dataframe(row, model_name)
            if layer_df.empty:
                patched_best_logit = _safe_float(row["neg_target_logit"])
                patched_best_prob = _safe_float(row["neg_target_prob"])
                patched_best_rank = int(row["neg_target_rank"])
                best_patch_layer = -1
                best_patch_type = "none"
                best_patch_delta = 0.0
            else:
                # pick strongest positive shift among FFN/Attn deltas
                best_ffn_idx = int(layer_df["ffn_delta"].idxmax())
                best_attn_idx = int(layer_df["attn_delta"].idxmax())
                best_ffn_val = float(pd.to_numeric(layer_df.loc[best_ffn_idx, "ffn_delta"], errors="coerce"))
                best_attn_val = float(pd.to_numeric(layer_df.loc[best_attn_idx, "attn_delta"], errors="coerce"))
                if best_ffn_val >= best_attn_val:
                    best_patch_type = "mlp"
                    best_patch_layer = int(pd.to_numeric(layer_df.loc[best_ffn_idx, "layer"], errors="coerce"))
                    best_patch_delta = best_ffn_val
                else:
                    best_patch_type = "attn"
                    best_patch_layer = int(pd.to_numeric(layer_df.loc[best_attn_idx, "layer"], errors="coerce"))
                    best_patch_delta = best_attn_val

                # Approximate patched values from negated baseline + best delta
                patched_best_logit = float(row["neg_target_logit"]) + best_patch_delta
                patched_best_prob = np.nan
                patched_best_rank = int(row["neg_target_rank"])

            case_metric = {
                "case_id": int(case_id),
                "subject": str(row["subject"]),
                "model_name": model_name,
                "positive_prompt": str(row["positive_prompt"]),
                "negated_prompt": str(row["negated_prompt"]),
                "target_token": str(row["target_token"]),
                "negation_failure": bool(row["negation_failure"]),
                "sgr": _safe_float(row["sgr"]),
                "pos_target_logit": _safe_float(row["pos_target_logit"]),
                "neg_target_logit": _safe_float(row["neg_target_logit"]),
                "patched_best_logit": patched_best_logit,
                "pos_target_prob": _safe_float(row["pos_target_prob"]),
                "neg_target_prob": _safe_float(row["neg_target_prob"]),
                "patched_best_prob": patched_best_prob,
                "pos_target_rank": int(row["pos_target_rank"]),
                "neg_target_rank": int(row["neg_target_rank"]),
                "patched_best_rank": patched_best_rank,
                "best_patch_type": best_patch_type,
                "best_patch_layer": int(best_patch_layer),
                "best_patch_delta": float(best_patch_delta),
                "crossover_layer": _safe_float(row.get("crossover_layer", np.nan)),
                "retrieval_strength": _safe_float(row.get("retrieval_strength", np.nan)),
                "inhibition_strength": _safe_float(row.get("inhibition_strength", np.nan)),
            }
            selected_rows.append(case_metric)
            case_metrics_rows.append(case_metric)

            if not layer_df.empty:
                layer_df = layer_df.copy()
                layer_df.insert(0, "case_id", int(case_id))
                case_layer_rows.append(layer_df)

        case_metrics_df = pd.DataFrame(case_metrics_rows)
        case_metrics_df.to_csv(case_results_dir / "case_metrics.csv", index=False)

        if case_layer_rows:
            case_layer_df = pd.concat(case_layer_rows, ignore_index=True)
            case_layer_df.to_csv(case_results_dir / "layer_metrics.csv", index=False)
            _plot_case_layer_lines(case_layer_df, case_figures_dir / "layer_behavior.png", case_id)
            layer_stats_df = _layer_concentration_stats(case_layer_df)
            layer_stats_df.insert(0, "case_id", int(case_id))
            layer_stats_df.to_csv(case_results_dir / "layer_concentration.csv", index=False)
            per_case_layer_stats[case_id] = layer_stats_df
            case_layer_frames.append(case_layer_df)
        else:
            per_case_layer_stats[case_id] = pd.DataFrame()

        _plot_case_behavior(case_metrics_df, case_figures_dir / "behavior_summary.png", case_id)

    selected_metrics_df = pd.DataFrame(selected_rows)
    selected_metrics_df.to_csv(results_output_dir / "qualitative_selected_case_metrics.csv", index=False)
    pd.DataFrame({"case_id": selected_case_ids}).to_csv(results_output_dir / "qualitative_selected_case_ids.csv", index=False)

    if case_layer_frames:
        case_layer_all_df = pd.concat(case_layer_frames, ignore_index=True)
        _make_global_figures(selected_metrics_df, case_layer_all_df, figures_output_dir)
    else:
        case_layer_all_df = pd.DataFrame()

    report_path = reports_output_dir / "qualitative.md"
    _build_markdown_report(
        output_path=report_path,
        figures_output_dir=figures_output_dir,
        results_output_dir=results_output_dir,
        selected_case_ids=selected_case_ids,
        selected_metrics_df=selected_metrics_df,
        per_case_layer_stats=per_case_layer_stats,
    )

    print(f"Saved report -> {report_path}")
    print(f"Saved selected case IDs -> {results_output_dir / 'qualitative_selected_case_ids.csv'}")
    print(f"Saved selected metrics -> {results_output_dir / 'qualitative_selected_case_metrics.csv'}")
    print(f"Saved case CSVs under -> {results_output_dir / 'qualitative_cases'}")
    print(f"Saved figures under -> {figures_output_dir}")


if __name__ == "__main__":
    main()
