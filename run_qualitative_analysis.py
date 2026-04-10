"""
Generate concise qualitative case studies across the project models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import activation_patching_scan, patched_prompt_metrics
from src.models import CANONICAL_MODEL_NAMES, load_model
from src.utils import benchmark_csv_path, load_benchmark_dataframe


torch.manual_seed(67)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate concise qualitative reports from benchmark CSVs")
    parser.add_argument("--models", nargs="+", default=CANONICAL_MODEL_NAMES, help="Models to include")
    parser.add_argument("--results_dir", default="results/cross_model", help="Directory containing per-model benchmark CSVs")
    parser.add_argument("--negator_suffix", default=" not", help="Negator suffix used in the benchmark CSVs")
    parser.add_argument("--examples_per_model", type=int, default=3, help="Representative cases per model")
    parser.add_argument("--top_k_predictions", type=int, default=5, help="Top-k next-token predictions to show")
    parser.add_argument("--output_md", default="reports/qualitative_analysis.md", help="Markdown report path")
    return parser.parse_args()


def _fmt_token(token: str) -> str:
    token = token.replace("\n", "\\n")
    return token if token else "<empty>"


def _prediction_rows(rows: list[dict[str, float]]) -> str:
    return " | ".join(f"`{_fmt_token(row['token'])}` ({row['prob']:.4f})" for row in rows)


def _top_predictions(model, prompt: str, target_token: str, top_k: int) -> dict:
    target_id = model.to_single_token(target_token)
    logits = model(model.to_tokens(prompt))
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, top_k)
    return {
        "target_logit": float(last_logits[target_id].item()),
        "target_prob": float(probs[target_id].item()),
        "target_rank": int((last_logits >= last_logits[target_id]).sum().item()),
        "top_predictions": [
            {"token": model.to_string(token_id), "prob": float(prob)}
            for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist())
        ],
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


def _case_reading(row: pd.Series, patch_scan: dict) -> str:
    neg_rank = int(row["neg_target_rank"])
    pos_rank = int(row["pos_target_rank"])
    best_patch = patch_scan["best_patch"]
    if bool(row["negation_failure"]):
        return (
            f"The target climbs after negation (rank {pos_rank} -> {neg_rank}), so this is a direct failure. "
            f"The strongest patch is {best_patch['patch_type']} at layer {best_patch['layer']} with Δ logit {best_patch['delta']:+.3f}."
        )
    if float(row["sgr"]) > 1:
        return (
            f"The model suppresses the target (rank {pos_rank} -> {neg_rank}), but SGR stays above 1. "
            f"This is a mismatch case and weakens any strict threshold interpretation."
        )
    return (
        f"The target is suppressed cleanly (rank {pos_rank} -> {neg_rank}) and the case agrees with the main story. "
        f"The best patch still appears at {best_patch['patch_type']} layer {best_patch['layer']}."
    )


def main():
    args = parse_args()
    output_lines = [
        "# Qualitative Analysis",
        "",
        "- Models: " + ", ".join(f"`{model}`" for model in args.models),
        f"- Results source: `{args.results_dir}`",
        f"- Negator suffix: `{args.negator_suffix}`",
        f"- Cases per model: `{args.examples_per_model}`",
        "",
    ]

    for model_name in args.models:
        csv_path = benchmark_csv_path(args.results_dir, model_name, args.negator_suffix)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing benchmark CSV for {model_name}: {csv_path}")

        df = load_benchmark_dataframe(csv_path)
        selected = _select_examples(df, args.examples_per_model)

        model = load_model(model_name)
        output_lines.extend([f"## {model_name}", ""])

        model_supports: list[str] = []
        model_hurts: list[str] = []

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

            reading = _case_reading(row, patch_scan)
            if bool(row["negation_failure"]) and best_patch["delta"] > 0:
                model_supports.append(
                    f"Case `{int(row['case_id'])}` is a true failure and still admits a positive patch effect ({best_patch['delta']:+.3f})."
                )
            if (not bool(row["negation_failure"])) and float(row["sgr"]) > 1:
                model_hurts.append(
                    f"Case `{int(row['case_id'])}` succeeds despite SGR `{float(row['sgr']):.3f}`, showing the metric is not decisive on its own."
                )

            output_lines.extend(
                [
                    f"### Case {int(row['case_id'])}: {row['subject']}",
                    "",
                    f"- Bucket: `{row['case_bucket']}`",
                    f"- Positive prompt: `{row['positive_prompt']}`",
                    f"- Negated prompt: `{row['negated_prompt']}`",
                    f"- Target token: `{row['target_token']}`",
                    f"- Outcome: `{'failure' if bool(row['negation_failure']) else 'success'}`",
                    f"- SGR: `{float(row['sgr']):.3f}`",
                    f"- Best patch: `{best_patch['patch_type']}` at layer `{best_patch['layer']}` with Δ logit `{best_patch['delta']:+.3f}`",
                    f"- Reading: {reading}",
                    "",
                    "| Run | Target logit | Target prob | Target rank |",
                    "|---|---:|---:|---:|",
                    f"| Positive | {positive['target_logit']:.3f} | {positive['target_prob']:.4f} | {positive['target_rank']} |",
                    f"| Negated | {negated['target_logit']:.3f} | {negated['target_prob']:.4f} | {negated['target_rank']} |",
                    f"| Negated (best patch) | {patched['target_logit']:.3f} | {patched['target_prob']:.4f} | {patched['target_rank']} |",
                    "",
                    f"- Positive top predictions: {_prediction_rows(positive['top_predictions'])}",
                    f"- Negated top predictions: {_prediction_rows(negated['top_predictions'])}",
                    f"- Patched top predictions: {_prediction_rows(patched['top_predictions'])}",
                    "",
                ]
            )

        output_lines.extend(["### What Helps The Story", ""])
        output_lines.extend([f"- {line}" for line in model_supports] or ["- No especially helpful qualitative case stood out in this sample."])
        output_lines.extend(["", "### What Hurts The Story", ""])
        output_lines.extend([f"- {line}" for line in model_hurts] or ["- No strong qualitative mismatch stood out in this sample."])
        output_lines.extend([""])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Saved qualitative markdown report -> {output_path}")


if __name__ == "__main__":
    main()
