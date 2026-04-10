"""
Generate a shared-case qualitative report across all selected models.
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
    parser = argparse.ArgumentParser(description="Multi-model qualitative report")
    parser.add_argument("--models", nargs="+", default=CANONICAL_MODEL_NAMES, help="Models to include in the report")
    parser.add_argument("--results_dir", default="results/cross_model", help="Directory containing per-model benchmark CSVs")
    parser.add_argument("--negator_suffix", default=" not", help="Negator suffix used in the benchmark CSVs")
    parser.add_argument("--case_ids", nargs="+", type=int, help="Optional fixed case_ids to analyse")
    parser.add_argument("--top_k_predictions", type=int, default=5, help="How many predictions to show per prompt")
    parser.add_argument("--output_md", default="reports/qualitative_multimodel_report.md", help="Markdown report path")
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


def main():
    args = parse_args()
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

    output_lines = [
        "# Multi-Model Qualitative Report",
        "",
        "- Models: " + ", ".join(f"`{model}`" for model in args.models),
        "- Shared case IDs: " + ", ".join(str(case_id) for case_id in selected_case_ids),
        "",
    ]

    helpful_lines: list[str] = []
    hurt_lines: list[str] = []

    analyses: dict[str, dict[int, dict]] = {}
    for model_name in args.models:
        print(f"Loading {model_name} for shared-case analysis...")
        model = load_model(model_name)
        analyses[model_name] = {}
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
            analyses[model_name][case_id] = {
                "row": row,
                "positive": positive,
                "negated": negated,
                "best_patch": best_patch,
                "patched": patched,
            }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for case_id in selected_case_ids:
        ref_row = None
        for model_name, df in data_by_model.items():
            candidate = df[df["case_id"] == case_id]
            if not candidate.empty:
                ref_row = candidate.iloc[0]
                break
        if ref_row is None:
            continue

        output_lines.extend(
            [
                f"## Case {case_id}: {ref_row['subject']}",
                "",
                f"- Positive prompt: `{ref_row['positive_prompt']}`",
                f"- Negated prompt: `{ref_row['negated_prompt']}`",
                f"- Target token: `{ref_row['target_token']}`",
                "",
            ]
        )

        labels = set()
        for model_name in args.models:
            case = analyses[model_name][case_id]
            row = case["row"]
            positive = case["positive"]
            negated = case["negated"]
            best_patch = case["best_patch"]
            patched = case["patched"]
            labels.add("failure" if bool(row["negation_failure"]) else "success")

            if bool(row["negation_failure"]) and patched["target_rank"] > negated["target_rank"]:
                helpful_lines.append(
                    f"Case `{case_id}` in `{model_name}` is a failure that improves under patching "
                    f"(rank {negated['target_rank']} -> {patched['target_rank']})."
                )
            if (not bool(row["negation_failure"])) and float(row["sgr"]) > 1:
                hurt_lines.append(
                    f"Case `{case_id}` in `{model_name}` succeeds even with SGR `{float(row['sgr']):.3f}`."
                )

            output_lines.extend(
                [
                    f"### {model_name}",
                    "",
                    f"- Outcome: `{'failure' if bool(row['negation_failure']) else 'success'}`",
                    f"- SGR: `{float(row['sgr']):.3f}`",
                    f"- Best patch: `{best_patch['patch_type']}` at layer `{best_patch['layer']}` with Δ logit `{best_patch['delta']:+.3f}`",
                    "",
                    "| Run | Target logit | Target prob | Target rank |",
                    "|---|---:|---:|---:|",
                    f"| Positive | {positive['target_logit']:.3f} | {positive['target_prob']:.4f} | {positive['target_rank']} |",
                    f"| Negated | {negated['target_logit']:.3f} | {negated['target_prob']:.4f} | {negated['target_rank']} |",
                    f"| Negated (best patch) | {patched['target_logit']:.3f} | {patched['target_prob']:.4f} | {patched['target_rank']} |",
                    "",
                    f"- Negated top predictions: {_prediction_rows(negated['top_predictions'])}",
                    f"- Patched top predictions: {_prediction_rows(patched['top_predictions'])}",
                    "",
                ]
            )

        if len(labels) > 1:
            hurt_lines.append(
                f"Case `{case_id}` does not behave consistently across models, which weakens any one-size-fits-all narrative."
            )

    output_lines.extend(["## What Helps The Story", ""])
    output_lines.extend([f"- {line}" for line in helpful_lines] or ["- No especially strong shared-case support stood out in this rerun."])
    output_lines.extend(["", "## What Hurts The Story", ""])
    output_lines.extend([f"- {line}" for line in hurt_lines] or ["- No especially strong shared-case contradiction stood out in this rerun."])
    output_lines.extend([""])

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Saved multi-model qualitative markdown report -> {output_path}")


if __name__ == "__main__":
    main()
