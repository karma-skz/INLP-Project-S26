"""
Generate metric-only qualitative analysis artifacts across selected models.
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
    parser = argparse.ArgumentParser(description="Generate metric-only qualitative analysis from benchmark CSVs")
    parser.add_argument("--models", nargs="+", default=CANONICAL_MODEL_NAMES, help="Models to include")
    parser.add_argument("--results_dir", default="results/cross_model", help="Directory containing per-model benchmark CSVs")
    parser.add_argument("--negator_suffix", default=" not", help="Negator suffix used in the benchmark CSVs")
    parser.add_argument("--examples_per_model", type=int, default=3, help="Representative cases per model")
    parser.add_argument("--top_k_predictions", type=int, default=5, help="Top-k next-token predictions to show")
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


def _select_examples(df: pd.DataFrame, examples_per_model: int) -> pd.DataFrame:
    work = df.copy()
    work["rank_improvement"] = work["pos_target_rank"] - work["neg_target_rank"]
    work["rank_shift"] = work["neg_target_rank"] - work["pos_target_rank"]
    work["distance_to_one"] = np.abs(work["sgr"].replace([np.inf, -np.inf], np.nan) - 1.0)

    selections = []
    seen = set()

    candidate_buckets = [
        ("largest_failure", work[work["negation_failure"]].sort_values(["rank_improvement", "sgr"], ascending=[False, False])),
        ("clean_success", work[~work["negation_failure"]].sort_values(["rank_shift", "sgr"], ascending=[False, True])),
        (
            "sgr_mismatch",
            work[
                ((~work["negation_failure"]) & (work["sgr"] > 1))
                | (work["negation_failure"] & (work["sgr"] <= 1))
            ].sort_values("distance_to_one", ascending=True),
        ),
    ]

    for label, bucket in candidate_buckets:
        for _, row in bucket.iterrows():
            case_id = int(row["case_id"])
            if case_id in seen:
                continue
            row = row.copy()
            row["case_bucket"] = label
            selections.append(row)
            seen.add(case_id)
            break

    if len(selections) < examples_per_model:
        remaining = work.sort_values(["negation_failure", "distance_to_one"], ascending=[False, True])
        for _, row in remaining.iterrows():
            case_id = int(row["case_id"])
            if case_id in seen:
                continue
            row = row.copy()
            row["case_bucket"] = "extra_case"
            selections.append(row)
            seen.add(case_id)
            if len(selections) >= examples_per_model:
                break

    return pd.DataFrame(selections[:examples_per_model])


def _save_metric_plots(summary_df: pd.DataFrame, case_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if not summary_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        model_names = summary_df["model_name"]

        axes[0].bar(model_names, summary_df["failure_rate"], color="#E45756")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("Failure rate")
        axes[0].set_ylabel("Rate")

        axes[1].bar(model_names, summary_df["sgr_mismatch_rate"], color="#4C78A8")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("SGR mismatch rate")

        axes[2].bar(model_names, summary_df["mean_best_patch_delta"], color="#54A24B")
        axes[2].axhline(0.0, color="black", linewidth=1)
        lo, hi = dynamic_axis_limits(summary_df["mean_best_patch_delta"])
        axes[2].set_ylim(lo, hi)
        axes[2].set_title("Mean best patch Δlogit")

        for ax in axes:
            ax.set_xlabel("Model")
            ax.tick_params(axis="x", rotation=25)

        fig.tight_layout()
        out = output_dir / "qualitative_model_summary.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)

    if not case_df.empty:
        plot_df = case_df.copy()
        plot_df["model_case"] = plot_df["model_name"] + "#" + plot_df["case_id"].astype(str)
        fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.55), 4.5))
        ax.bar(plot_df["model_case"], plot_df["best_patch_delta"], color="#B279A2")
        ax.axhline(0.0, color="black", linewidth=1)
        lo, hi = dynamic_axis_limits(plot_df["best_patch_delta"])
        ax.set_ylim(lo, hi)
        ax.set_title("Best patch Δlogit by selected case")
        ax.set_xlabel("Model#Case")
        ax.set_ylabel("Δlogit")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        out = output_dir / "qualitative_case_patch_delta.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)

    return paths


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_summary_rows: list[dict[str, object]] = []
    case_metric_rows: list[dict[str, object]] = []

    for model_name in args.models:
        csv_path = benchmark_csv_path(args.results_dir, model_name, args.negator_suffix)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing benchmark CSV for {model_name}: {csv_path}")

        df = load_benchmark_dataframe(csv_path)
        selected = _select_examples(df, args.examples_per_model)

        model = load_model(model_name)

        model_case_metrics: list[dict[str, object]] = []

        for _, row in selected.iterrows():
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

            case_metrics = {
                "model_name": model_name,
                "case_id": int(row["case_id"]),
                "bucket": row["case_bucket"],
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
            model_case_metrics.append(case_metrics)
            case_metric_rows.append(case_metrics)

        model_case_df = pd.DataFrame(model_case_metrics)
        mismatch_rate = float(
            (
                ((~model_case_df["negation_failure"]) & (model_case_df["sgr"] > 1))
                | ((model_case_df["negation_failure"]) & (model_case_df["sgr"] <= 1))
            ).mean()
        ) if not model_case_df.empty else float("nan")
        patch_harm_rate = float((model_case_df["rank_change_after_patch"] < 0).mean()) if not model_case_df.empty else float("nan")

        model_summary_rows.append(
            {
                "model_name": model_name,
                "n_cases": int(len(model_case_df)),
                "failure_rate": float(model_case_df["negation_failure"].mean()) if not model_case_df.empty else float("nan"),
                "sgr_mismatch_rate": mismatch_rate,
                "mean_best_patch_delta": float(model_case_df["best_patch_delta"].mean()) if not model_case_df.empty else float("nan"),
                "patch_harm_rate": patch_harm_rate,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(model_summary_rows)
    case_df = pd.DataFrame(case_metric_rows)
    summary_csv_path = output_dir / "qualitative_metrics_summary.csv"
    case_csv_path = output_dir / "qualitative_case_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    case_df.to_csv(case_csv_path, index=False)
    plot_paths = _save_metric_plots(summary_df, case_df, output_dir)

    print(f"Saved summary metrics CSV -> {summary_csv_path}")
    print(f"Saved case-level metrics CSV -> {case_csv_path}")
    for plot_path in plot_paths:
        print(f"Saved metric plot -> {plot_path}")


if __name__ == "__main__":
    main()
