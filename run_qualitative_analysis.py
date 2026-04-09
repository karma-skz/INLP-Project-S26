"""
run_qualitative_analysis.py
===========================
Generate a qualitative case-study report from an existing benchmark CSV.

The script selects representative examples from the benchmark output,
reruns those prompt pairs through the model, and writes a markdown report
with:
  - behavioural comparison (target rank/logit/prob, top-k predictions)
  - component-level DLA differences
  - per-layer retrieval vs inhibition summaries
  - top inhibition heads and a per-head heatmap
  - optional activation-patching summaries

Example
-------
    # Build a qualitative report from the GPT-2 benchmark
    python run_qualitative_analysis.py

    # Analyse specific cases only
    python run_qualitative_analysis.py --case_ids 1 4 57

    # Include activation patching for a smaller set of cases
    python run_qualitative_analysis.py --case_ids 1 4 --with_patching
"""

from __future__ import annotations

import argparse
import os
import re
from functools import partial
from pathlib import Path
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from src.analysis.per_head import per_head_dla, plot_head_dla_heatmap, top_inhibition_heads
from src.models import load_model
from src.utils import load_benchmark_dataframe


torch.manual_seed(67)


def parse_args():
    p = argparse.ArgumentParser(description="Generate qualitative case studies from benchmark results")
    p.add_argument(
        "--results_csv",
        default="results/gpt2-small_not_benchmark.csv",
        help="Benchmark CSV to sample cases from",
    )
    p.add_argument(
        "--model",
        default="gpt2-small",
        choices=["gpt2-small", "gpt2", "gpt2-medium", "gpt2-large", "pythia-160m", "pythia", "pythia-410m"],
        help="Model to load for rerunning qualitative examples",
    )
    p.add_argument(
        "--case_ids",
        nargs="+",
        type=int,
        help="Specific case_ids to analyse; if omitted, representative cases are auto-selected",
    )
    p.add_argument(
        "--examples_per_bucket",
        type=int,
        default=2,
        help="How many cases to select from each qualitative bucket when --case_ids is omitted",
    )
    p.add_argument(
        "--top_k_predictions",
        type=int,
        default=8,
        help="Top-k next-token predictions to show for each prompt",
    )
    p.add_argument(
        "--top_k_components",
        type=int,
        default=8,
        help="How many DLA components to list per case",
    )
    p.add_argument(
        "--top_k_heads",
        type=int,
        default=8,
        help="How many inhibition heads to list per case",
    )
    p.add_argument(
        "--output_md",
        default="reports/qualitative_analysis.md",
        help="Markdown report path",
    )
    p.add_argument(
        "--fig_dir",
        default="figures/qualitative",
        help="Directory for per-case qualitative figures",
    )
    p.add_argument(
        "--with_patching",
        action="store_true",
        help="Run activation patching summaries for each selected case (slower)",
    )
    return p.parse_args()


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return value.strip("-") or "case"


def _fmt_token(token: str) -> str:
    token = token.replace("\n", "\\n")
    if token == "":
        return "<empty>"
    return repr(token)


def _top_predictions(model, prompt: str, target_id: int, k: int) -> dict:
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, k)
    top_rows = []
    for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        top_rows.append({
            "token": model.to_string(token_id),
            "prob": float(prob),
        })
    return {
        "logits": last_logits,
        "target_logit": float(last_logits[target_id].item()),
        "target_prob": float(probs[target_id].item()),
        "target_rank": int((last_logits >= last_logits[target_id]).sum().item()),
        "top_predictions": top_rows,
    }


def _per_layer_dla(labels: List[str], pos_dla: np.ndarray, neg_dla: np.ndarray, n_layers: int):
    ffn_pos = np.zeros(n_layers)
    ffn_neg = np.zeros(n_layers)
    attn_pos = np.zeros(n_layers)
    attn_neg = np.zeros(n_layers)

    for idx, label in enumerate(labels):
        parts = label.split("_")
        if len(parts) >= 2 and parts[0].isdigit():
            layer = int(parts[0])
            if "mlp" in label.lower():
                ffn_pos[layer] += pos_dla[idx]
                ffn_neg[layer] += neg_dla[idx]
            elif "attn" in label.lower():
                attn_pos[layer] += pos_dla[idx]
                attn_neg[layer] += neg_dla[idx]

    return ffn_pos, ffn_neg, attn_pos, attn_neg


def _select_examples(df: pd.DataFrame, examples_per_bucket: int) -> pd.DataFrame:
    work = df.copy()
    work["rank_improvement"] = work["pos_target_rank"] - work["neg_target_rank"]
    work["suppression_gain"] = work["neg_target_rank"] - work["pos_target_rank"]

    selections = []
    seen_case_ids = set()

    buckets = [
        work[work["negation_failure"]].sort_values(["rank_improvement", "sgr"], ascending=[False, False]),
        work[~work["negation_failure"]].sort_values(["suppression_gain", "sgr"], ascending=[False, True]),
        work[np.isfinite(work["sgr"])].sort_values("sgr", ascending=True),
    ]

    for bucket in buckets:
        taken = 0
        for _, row in bucket.iterrows():
            case_id = int(row["case_id"])
            if case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            selections.append(row)
            taken += 1
            if taken >= examples_per_bucket:
                break

    return pd.DataFrame(selections)


def _patch_last_token(value, hook, pos_cache):
    value[:, -1, :] = pos_cache[hook.name][:, -1, :]
    return value


@torch.no_grad()
def _activation_patching_summary(model, positive_prompt: str, negated_prompt: str, target_token: str) -> dict:
    target_id = model.to_single_token(target_token)
    n_layers = model.cfg.n_layers

    _, pos_cache = model.run_with_cache(positive_prompt)
    baseline_logits = model(model.to_tokens(negated_prompt))
    baseline_logit = float(baseline_logits[0, -1, target_id].item())

    patch_sets = {
        "resid": "hook_resid_post",
        "mlp": "hook_mlp_out",
        "attn": "hook_attn_out",
    }
    out = {"baseline_logit": baseline_logit}

    for label, hook_suffix in patch_sets.items():
        deltas = []
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.{hook_suffix}"
            patched_logits = model.run_with_hooks(
                negated_prompt,
                fwd_hooks=[(hook_name, partial(_patch_last_token, pos_cache=pos_cache))],
            )
            patched_logit = float(patched_logits[0, -1, target_id].item())
            deltas.append(patched_logit - baseline_logit)
        out[label] = deltas

    return out


def _markdown_prediction_table(title: str, rows: Iterable[dict]) -> List[str]:
    lines = [f"**{title}**", "", "| Token | Prob |", "|---|---:|"]
    for row in rows:
        lines.append(f"| `{_fmt_token(row['token'])}` | {row['prob']:.4f} |")
    lines.append("")
    return lines


def _interpret_case(pos_rank: int, neg_rank: int, sgr: float) -> str:
    if neg_rank < pos_rank:
        return (
            f"The target becomes more likely after negation (rank {pos_rank} -> {neg_rank}), "
            f"which is a direct negation failure. The SGR of {sgr:.3f} suggests retrieval still dominates inhibition."
        )
    if np.isfinite(sgr) and sgr < 1:
        return (
            f"The negated prompt suppresses the target (rank {pos_rank} -> {neg_rank}) and SGR falls below 1, "
            "which matches the intended gating story."
        )
    return (
        f"The negated prompt suppresses the target (rank {pos_rank} -> {neg_rank}), "
        f"but the SGR remains {sgr:.3f}; this is a useful mismatch case for the metric."
    )


@torch.no_grad()
def analyse_case(model, row: pd.Series, args) -> dict:
    target_id = model.to_single_token(row["target_token"])

    pos_info = _top_predictions(model, row["positive_prompt"], target_id, args.top_k_predictions)
    neg_info = _top_predictions(model, row["negated_prompt"], target_id, args.top_k_predictions)

    _, pos_cache = model.run_with_cache(row["positive_prompt"])
    _, neg_cache = model.run_with_cache(row["negated_prompt"])

    pos_resid, labels = pos_cache.decompose_resid(return_labels=True, mode="full")
    neg_resid, _ = neg_cache.decompose_resid(return_labels=True, mode="full")

    target_unembed = model.W_U[:, target_id]
    pos_dla = (pos_resid[:, 0, -1, :] @ target_unembed).detach().cpu().numpy()
    neg_dla = (neg_resid[:, 0, -1, :] @ target_unembed).detach().cpu().numpy()
    delta_dla = neg_dla - pos_dla

    top_component_idx = np.argsort(np.abs(delta_dla))[::-1][:args.top_k_components]
    top_components = [
        {
            "label": labels[i],
            "pos_dla": float(pos_dla[i]),
            "neg_dla": float(neg_dla[i]),
            "delta": float(delta_dla[i]),
        }
        for i in top_component_idx
    ]

    ffn_pos, ffn_neg, attn_pos, attn_neg = _per_layer_dla(labels, pos_dla, neg_dla, model.cfg.n_layers)
    top_retrieval_layers = np.argsort(ffn_neg)[::-1][:3].tolist()
    top_inhibition_layers = np.argsort(attn_neg)[:3].tolist()

    head_dla_pos, head_dla_neg = per_head_dla(
        model,
        row["positive_prompt"],
        row["negated_prompt"],
        row["target_token"],
    )
    top_heads = top_inhibition_heads(head_dla_pos, head_dla_neg, top_k=args.top_k_heads)

    slug = _slugify(f"{row['case_id']}-{row['subject']}")
    heatmap_filename = f"{slug}_head_dla_heatmap.png"
    plot_head_dla_heatmap(
        head_dla_pos,
        head_dla_neg,
        target_token=row["target_token"],
        fig_dir=args.fig_dir,
        filename=heatmap_filename,
    )

    patching = None
    if args.with_patching:
        patching = _activation_patching_summary(
            model,
            row["positive_prompt"],
            row["negated_prompt"],
            row["target_token"],
        )

    return {
        "row": row,
        "pos_info": pos_info,
        "neg_info": neg_info,
        "top_components": top_components,
        "ffn_neg": ffn_neg,
        "attn_neg": attn_neg,
        "top_retrieval_layers": top_retrieval_layers,
        "top_inhibition_layers": top_inhibition_layers,
        "top_heads": top_heads,
        "heatmap_path": os.path.join(args.fig_dir, heatmap_filename),
        "patching": patching,
    }


def build_report(cases: List[dict], args):
    lines: List[str] = []
    lines.append("# Qualitative Analysis")
    lines.append("")
    lines.append(f"- Results source: `{args.results_csv}`")
    lines.append(f"- Model rerun: `{args.model}`")
    lines.append(f"- Cases analysed: {len(cases)}")
    lines.append("")

    for case in cases:
        row = case["row"]
        pos_info = case["pos_info"]
        neg_info = case["neg_info"]
        lines.append(f"## Case {int(row['case_id'])}: {row['subject']}")
        lines.append("")
        lines.append(f"- Positive prompt: `{row['positive_prompt']}`")
        lines.append(f"- Negated prompt: `{row['negated_prompt']}`")
        lines.append(f"- Target token: `{row['target_token']}`")
        lines.append(f"- Benchmark label: `{'failure' if bool(row['negation_failure']) else 'success'}`")
        lines.append(f"- SGR: `{float(row['sgr']):.3f}`")
        lines.append("")
        lines.append(_interpret_case(pos_info["target_rank"], neg_info["target_rank"], float(row["sgr"])))
        lines.append("")
        lines.append("| Metric | Positive | Negated |")
        lines.append("|---|---:|---:|")
        lines.append(f"| Target logit | {pos_info['target_logit']:.3f} | {neg_info['target_logit']:.3f} |")
        lines.append(f"| Target prob | {pos_info['target_prob']:.4f} | {neg_info['target_prob']:.4f} |")
        lines.append(f"| Target rank | {pos_info['target_rank']} | {neg_info['target_rank']} |")
        lines.append("")

        lines.extend(_markdown_prediction_table("Top Predictions: Positive", pos_info["top_predictions"]))
        lines.extend(_markdown_prediction_table("Top Predictions: Negated", neg_info["top_predictions"]))

        lines.append("**Largest Component Shifts (Negated - Positive DLA)**")
        lines.append("")
        lines.append("| Component | Pos DLA | Neg DLA | Delta |")
        lines.append("|---|---:|---:|---:|")
        for item in case["top_components"]:
            lines.append(
                f"| `{item['label']}` | {item['pos_dla']:.3f} | {item['neg_dla']:.3f} | {item['delta']:.3f} |"
            )
        lines.append("")

        lines.append("**Per-Layer Summary on the Negated Prompt**")
        lines.append("")
        retrieval_summary = ", ".join(
            f"L{layer} ({case['ffn_neg'][layer]:.3f})"
            for layer in case["top_retrieval_layers"]
        )
        inhibition_summary = ", ".join(
            f"L{layer} ({case['attn_neg'][layer]:.3f})"
            for layer in case["top_inhibition_layers"]
        )
        lines.append(
            f"- Strongest retrieval layers (FFN DLA): {retrieval_summary}"
        )
        lines.append(
            f"- Strongest inhibition layers (Attn DLA): {inhibition_summary}"
        )
        lines.append("")

        lines.append("**Top Inhibition Heads**")
        lines.append("")
        lines.append("| Head | Delta DLA (pos - neg) |")
        lines.append("|---|---:|")
        for (layer, head), delta in case["top_heads"]:
            lines.append(f"| `({layer}, {head})` | {delta:.3f} |")
        lines.append("")
        lines.append(f"- Head heatmap: `{case['heatmap_path']}`")
        lines.append("")

        if case["patching"] is not None:
            patching = case["patching"]
            for label in ("resid", "mlp", "attn"):
                deltas = np.array(patching[label])
                top_idx = np.argsort(np.abs(deltas))[::-1][:3]
                summary = ", ".join(f"L{idx} ({deltas[idx]:+.3f})" for idx in top_idx)
                lines.append(f"- Strongest {label} patching layers: {summary}")
            lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_benchmark_dataframe(args.results_csv)
    if args.case_ids:
        selected = df[df["case_id"].isin(args.case_ids)].copy()
        missing = sorted(set(args.case_ids) - set(selected["case_id"].tolist()))
        if missing:
            print(f"Warning: case_ids not found in CSV: {missing}")
    else:
        selected = _select_examples(df, args.examples_per_bucket)

    if selected.empty:
        raise ValueError("No cases selected for qualitative analysis.")

    selected = selected.drop_duplicates(subset=["case_id"]).reset_index(drop=True)
    print(f"Selected {len(selected)} cases for qualitative analysis:")
    for _, row in selected.iterrows():
        print(
            f"  case_id={int(row['case_id']):>4}  "
            f"failure={bool(row['negation_failure'])!s:<5}  "
            f"sgr={float(row['sgr']):>8.3f}  "
            f"subject={row['subject']}"
        )

    model = load_model(args.model)

    cases = []
    for _, row in selected.iterrows():
        print(f"\nAnalysing case {int(row['case_id'])}: {row['subject']}")
        cases.append(analyse_case(model, row, args))

    report = build_report(cases, args)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nSaved qualitative report -> {output_path}")


if __name__ == "__main__":
    main()
