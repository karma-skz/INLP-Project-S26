"""
run_multi_model_qualitative_report.py
=====================================
Generate a standalone markdown report for qualitative case studies across
multiple models, comparing behaviour before and after activation patching.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import activation_patching_scan, patched_prompt_metrics
from src.models import load_model


torch.manual_seed(67)


DEFAULT_MODELS = ["gpt2-small", "pythia-160m"]


def parse_args():
    p = argparse.ArgumentParser(description="Multi-model qualitative report with activation patching")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to include in the report")
    p.add_argument("--results_dir", default="results/cross_model", help="Directory containing per-model benchmark CSVs")
    p.add_argument("--negator_suffix", default="not", help="Filename-safe negator suffix used in benchmark CSVs")
    p.add_argument("--case_ids", nargs="+", type=int, help="Optional fixed case_ids to analyse")
    p.add_argument("--output_md", default="reports/qualitative_multimodel_report.md", help="Markdown report path")
    p.add_argument("--top_k_predictions", type=int, default=5, help="How many predictions to show per prompt")
    return p.parse_args()


def _benchmark_path(results_dir: str, model_name: str, safe_suffix: str) -> Path:
    return Path(results_dir) / f"{model_name}_{safe_suffix}_benchmark.csv"


def _fmt_token(token: str) -> str:
    token = token.replace("\n", "\\n")
    return token if token else "<empty>"


def _prediction_rows(rows):
    return " | ".join(f"`{_fmt_token(row['token'])}` ({row['prob']:.4f})" for row in rows)


def _select_case_ids(primary_df: pd.DataFrame) -> list[int]:
    work = primary_df.copy()
    work["rank_improvement"] = work["pos_target_rank"] - work["neg_target_rank"]
    work["suppression_gain"] = work["neg_target_rank"] - work["pos_target_rank"]
    work["distance_to_one"] = np.abs(work["sgr"].replace([float("inf"), -float("inf")], np.nan) - 1.0)

    seen = set()
    selected = []

    def add_rows(df, n):
        taken = 0
        for _, row in df.iterrows():
            case_id = int(row["case_id"])
            if case_id in seen:
                continue
            seen.add(case_id)
            selected.append(case_id)
            taken += 1
            if taken >= n:
                break

    add_rows(work[work["negation_failure"]].sort_values(["rank_improvement", "sgr"], ascending=[False, False]), 2)
    add_rows(work[~work["negation_failure"]].sort_values(["suppression_gain", "distance_to_one"], ascending=[False, True]), 1)
    add_rows(work[(~work["negation_failure"]) & work["distance_to_one"].notna()].sort_values("distance_to_one", ascending=True), 1)
    return selected


def _top_predictions(model, prompt: str, target_token: str, top_k: int):
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


def _load_results(models: list[str], results_dir: str, safe_suffix: str):
    data = {}
    for model_name in models:
        path = _benchmark_path(results_dir, model_name, safe_suffix)
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark CSV for {model_name}: {path}")
        data[model_name] = pd.read_csv(path)
    return data


def build_report(selected_case_ids: list[int], rows_by_model: dict[str, pd.DataFrame], analyses: dict, output_path: Path):
    lines = [
        "# Multi-Model Qualitative Report",
        "",
        "- Goal: compare representative negation cases before and after activation patching across all tested models.",
        f"- Models: {', '.join(f'`{m}`' for m in rows_by_model)}",
        f"- Cases: {', '.join(str(cid) for cid in selected_case_ids)}",
        "",
    ]

    for case_id in selected_case_ids:
        ref_row = None
        for model_name in rows_by_model:
            candidate = rows_by_model[model_name]
            candidate = candidate[candidate["case_id"] == case_id]
            if not candidate.empty:
                ref_row = candidate.iloc[0]
                break
        if ref_row is None:
            continue

        lines.extend([
            f"## Case {case_id}: {ref_row['subject']}",
            "",
            f"- Positive prompt: `{ref_row['positive_prompt']}`",
            f"- Negated prompt: `{ref_row['negated_prompt']}`",
            f"- Target token: `{ref_row['target_token']}`",
            "",
        ])

        for model_name in rows_by_model:
            case = analyses[model_name][case_id]
            row = case["row"]
            scan = case["patch_scan"]
            best = scan["best_patch"]
            lines.extend([
                f"### {model_name}",
                "",
                f"- Benchmark label: `{'failure' if bool(row['negation_failure']) else 'success'}`",
                f"- SGR: `{float(row['sgr']):.3f}`",
                f"- Best patch: `{best['patch_type']}` at layer `{best['layer']}` with Δ logit `{best['delta']:+.3f}`",
                "",
                "| Run | Target logit | Target prob | Target rank |",
                "|---|---:|---:|---:|",
                f"| Positive | {case['positive']['target_logit']:.3f} | {case['positive']['target_prob']:.4f} | {case['positive']['target_rank']} |",
                f"| Negated (unpatched) | {scan['baseline']['target_logit']:.3f} | {scan['baseline']['target_prob']:.4f} | {scan['baseline']['target_rank']} |",
                f"| Negated (best patch) | {case['patched']['target_logit']:.3f} | {case['patched']['target_prob']:.4f} | {case['patched']['target_rank']} |",
                "",
                f"- Negated top predictions: {_prediction_rows(scan['baseline']['top_predictions'])}",
                f"- Patched top predictions: {_prediction_rows(case['patched']['top_predictions'])}",
                f"- Strongest residual patch layer: `L{scan['patches']['resid']['best_layer']}` (Δ `{scan['patches']['resid']['best_delta']:+.3f}`)",
                f"- Strongest MLP patch layer: `L{scan['patches']['mlp']['best_layer']}` (Δ `{scan['patches']['mlp']['best_delta']:+.3f}`)",
                f"- Strongest attention patch layer: `L{scan['patches']['attn']['best_layer']}` (Δ `{scan['patches']['attn']['best_delta']:+.3f}`)",
                "",
            ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    safe_suffix = args.negator_suffix.strip().replace(" ", "-")
    data_by_model = _load_results(args.models, args.results_dir, safe_suffix)

    common_case_ids = set.intersection(*(set(df["case_id"].tolist()) for df in data_by_model.values()))
    if args.case_ids:
        selected_case_ids = [case_id for case_id in args.case_ids if case_id in common_case_ids]
        missing = sorted(set(args.case_ids) - set(selected_case_ids))
        if missing:
            print(f"Skipping case_ids missing from at least one model: {missing}")
    else:
        primary_df = data_by_model[args.models[0]]
        primary_df = primary_df[primary_df["case_id"].isin(common_case_ids)].copy()
        selected_case_ids = _select_case_ids(primary_df)

    if not selected_case_ids:
        raise ValueError("No shared case_ids available for qualitative analysis.")

    print(f"Selected shared case_ids: {selected_case_ids}")

    analyses = {}
    rows_by_model = {}
    for model_name in args.models:
        print(f"\nLoading {model_name} for qualitative analysis...")
        model = load_model(model_name)
        model_rows = data_by_model[model_name]
        model_rows = model_rows[model_rows["case_id"].isin(selected_case_ids)].copy()
        rows_by_model[model_name] = model_rows
        analyses[model_name] = {}

        for _, row in model_rows.iterrows():
            case_id = int(row["case_id"])
            print(f"  Analysing case {case_id}: {row['subject']}")
            positive = _top_predictions(model, row["positive_prompt"], row["target_token"], args.top_k_predictions)
            patch_scan = activation_patching_scan(
                model,
                row["positive_prompt"],
                row["negated_prompt"],
                row["target_token"],
                top_k=args.top_k_predictions,
            )
            best = patch_scan["best_patch"]
            patched = patched_prompt_metrics(
                model,
                row["positive_prompt"],
                row["negated_prompt"],
                row["target_token"],
                patch_type=best["patch_type"],
                layer=best["layer"],
                top_k=args.top_k_predictions,
            )
            analyses[model_name][case_id] = {
                "row": row,
                "positive": positive,
                "patch_scan": patch_scan,
                "patched": patched,
            }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(args.output_md)
    build_report(selected_case_ids, rows_by_model, analyses, output_path)
    print(f"\nSaved qualitative markdown report -> {output_path}")


if __name__ == "__main__":
    main()
