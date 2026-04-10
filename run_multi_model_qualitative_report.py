"""
Generate shared-case metric artifacts across selected models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.analysis import activation_patching_scan, patched_prompt_metrics
from src.models import CANONICAL_MODEL_NAMES, load_model
from src.utils import benchmark_csv_path, dynamic_axis_limits, load_benchmark_dataframe


torch.manual_seed(67)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model shared-case metric analysis")
    parser.add_argument("--models", nargs="+", default=CANONICAL_MODEL_NAMES, help="Models to include in the report")
    parser.add_argument("--results_dir", default="results/cross_model", help="Directory containing per-model benchmark CSVs")
    parser.add_argument("--negator_suffix", default=" not", help="Negator suffix used in the benchmark CSVs")
    parser.add_argument("--case_ids", nargs="+", type=int, help="Optional fixed case_ids to analyse")
    parser.add_argument("--top_k_predictions", type=int, default=5, help="How many predictions to show per prompt")
    parser.add_argument("--output_dir", default="reports", help="Output directory for metric CSVs and plots")
    return parser.parse_args()


def _top_predictions(model, prompt: str, target_token: str, top_k: int) -> dict:
    del top_k
    target_id = model.to_single_token(target_token)
    logits = model(model.to_tokens(prompt))
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    return {
        "target_logit": float(last_logits[target_id].item()),
        "target_prob": float(probs[target_id].item()),
        "target_rank": int((last_logits >= last_logits[target_id]).sum().item()),
    }


def _load_results(models: list[str], results_dir: str, negator_suffix: str) -> dict[str, pd.DataFrame]:
    data = {}
    for model_name in models:
        path = benchmark_csv_path(results_dir, model_name, negator_suffix)
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark CSV for {model_name}: {path}")
        data[model_name] = load_benchmark_dataframe(path)
    return data


def _select_case_ids(primary_df: pd.DataFrame) -> list[int]:
    work = primary_df.copy()
    work["rank_improvement"] = work["pos_target_rank"] - work["neg_target_rank"]
    work["rank_shift"] = work["neg_target_rank"] - work["pos_target_rank"]
    work["distance_to_one"] = np.abs(work["sgr"].replace([np.inf, -np.inf], np.nan) - 1.0)

    selected: list[int] = []
    seen: set[int] = set()
    buckets = [
        work[work["negation_failure"]].sort_values(["rank_improvement", "sgr"], ascending=[False, False]),
        work[~work["negation_failure"]].sort_values(["rank_shift", "sgr"], ascending=[False, True]),
        work[
            ((~work["negation_failure"]) & (work["sgr"] > 1))
            | (work["negation_failure"] & (work["sgr"] <= 1))
        ].sort_values("distance_to_one", ascending=True),
    ]

    for bucket in buckets:
        for _, row in bucket.iterrows():
            case_id = int(row["case_id"])
            if case_id in seen:
                continue
            selected.append(case_id)
            seen.add(case_id)
            break

    return selected


def _save_metric_plots(summary_df: pd.DataFrame, metrics_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if not summary_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        models = summary_df["model_name"]

        axes[0].bar(models, summary_df["failure_rate"], color="#E45756")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("Failure rate")
        axes[0].set_ylabel("Rate")

        axes[1].bar(models, summary_df["patch_harm_rate"], color="#F58518")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Patch harm rate")

        axes[2].bar(models, summary_df["mean_best_patch_delta"], color="#54A24B")
        axes[2].axhline(0.0, color="black", linewidth=1)
        lo, hi = dynamic_axis_limits(summary_df["mean_best_patch_delta"])
        axes[2].set_ylim(lo, hi)
        axes[2].set_title("Mean best patch Δlogit")

        for ax in axes:
            ax.set_xlabel("Model")
            ax.tick_params(axis="x", rotation=25)

        fig.tight_layout()
        out = output_dir / "qualitative_multimodel_summary.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)

    if not metrics_df.empty:
        plot_df = metrics_df.copy().sort_values(["model_name", "case_id"]).reset_index(drop=True)
        plot_df["model_case"] = plot_df["model_name"] + "#" + plot_df["case_id"].astype(str)
        fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.55), 4.5))
        ax.bar(plot_df["model_case"], plot_df["best_patch_delta"], color="#4C78A8")
        ax.axhline(0.0, color="black", linewidth=1)
        lo, hi = dynamic_axis_limits(plot_df["best_patch_delta"])
        ax.set_ylim(lo, hi)
        ax.set_title("Best patch Δlogit by model and case")
        ax.set_xlabel("Model#Case")
        ax.set_ylabel("Δlogit")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        out = output_dir / "qualitative_multimodel_case_patch_delta.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)

    return paths


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_by_model = _load_results(args.models, args.results_dir, args.negator_suffix)

    common_case_ids = set.intersection(*(set(df["case_id"].tolist()) for df in data_by_model.values()))
    if args.case_ids:
        selected_case_ids = [case_id for case_id in args.case_ids if case_id in common_case_ids]
    else:
        primary_df = data_by_model[args.models[0]]
        primary_df = primary_df[primary_df["case_id"].isin(common_case_ids)].copy()
        selected_case_ids = _select_case_ids(primary_df)

    if not selected_case_ids:
        raise ValueError("No shared case_ids available for qualitative analysis.")

    metric_rows: list[dict[str, object]] = []

    for model_name in args.models:
        print(f"Loading {model_name} for shared-case analysis...")
        model = load_model(model_name)
        for case_id in selected_case_ids:
            row = data_by_model[model_name][data_by_model[model_name]["case_id"] == case_id].iloc[0]
            positive = _top_predictions(model, row["positive_prompt"], row["target_token"], args.top_k_predictions)
            negated = _top_predictions(model, row["negated_prompt"], row["target_token"], args.top_k_predictions)
            patch_scan = activation_patching_scan(
                model,
                row["positive_prompt"],
                row["negated_prompt"],
                row["target_token"],
                top_k=args.top_k_predictions,
            )
            best_patch = patch_scan["best_patch"]
            patched = patched_prompt_metrics(
                model,
                row["positive_prompt"],
                row["negated_prompt"],
                row["target_token"],
                patch_type=best_patch["patch_type"],
                layer=best_patch["layer"],
                top_k=args.top_k_predictions,
            )

            metric_rows.append(
                {
                    "model_name": model_name,
                    "case_id": int(case_id),
                    "negation_failure": bool(row["negation_failure"]),
                    "sgr": float(row["sgr"]),
                    "best_patch_delta": float(best_patch["delta"]),
                    "best_patch_layer": int(best_patch["layer"]),
                    "best_patch_type": str(best_patch["patch_type"]),
                    "pos_target_logit": float(positive["target_logit"]),
                    "neg_target_logit": float(negated["target_logit"]),
                    "patched_target_logit": float(patched["target_logit"]),
                    "pos_target_prob": float(positive["target_prob"]),
                    "neg_target_prob": float(negated["target_prob"]),
                    "patched_target_prob": float(patched["target_prob"]),
                    "pos_rank": int(positive["target_rank"]),
                    "neg_rank": int(negated["target_rank"]),
                    "patched_rank": int(patched["target_rank"]),
                    "rank_change_after_patch": int(patched["target_rank"] - negated["target_rank"]),
                }
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metric_rows)
    summary_df = (
        metrics_df.groupby("model_name", as_index=False)
        .agg(
            n_cases=("case_id", "count"),
            failure_rate=("negation_failure", "mean"),
            sgr_mismatch_rate=("sgr", lambda s: float((((metrics_df.loc[s.index, "negation_failure"] == False) & (s > 1)) | ((metrics_df.loc[s.index, "negation_failure"] == True) & (s <= 1))).mean())),
            mean_best_patch_delta=("best_patch_delta", "mean"),
            patch_harm_rate=("rank_change_after_patch", lambda s: float((s < 0).mean())),
        )
        .sort_values("model_name")
        .reset_index(drop=True)
    )

    selected_case_ids_df = pd.DataFrame({"case_id": selected_case_ids})

    summary_csv_path = output_dir / "qualitative_multimodel_metrics_summary.csv"
    case_csv_path = output_dir / "qualitative_multimodel_case_metrics.csv"
    selected_cases_csv_path = output_dir / "qualitative_multimodel_selected_case_ids.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    metrics_df.to_csv(case_csv_path, index=False)
    selected_case_ids_df.to_csv(selected_cases_csv_path, index=False)
    plot_paths = _save_metric_plots(summary_df, metrics_df, output_dir)

    print(f"Saved summary metrics CSV -> {summary_csv_path}")
    print(f"Saved case-level metrics CSV -> {case_csv_path}")
    print(f"Saved selected case IDs CSV -> {selected_cases_csv_path}")
    for plot_path in plot_paths:
        print(f"Saved metric plot -> {plot_path}")


if __name__ == "__main__":
    main()
