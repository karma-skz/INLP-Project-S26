from __future__ import annotations

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch

from src.analysis import EmotionDirectionResult, analyze_emotion_negation
from src.dataset import build_emotion_prompt_dataset
from src.models import CANONICAL_MODEL_NAMES, MODEL_SHORTNAMES, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run the emotion-negation directional analysis experiment")
    parser.add_argument(
        "--models",
        nargs="+",
        default=CANONICAL_MODEL_NAMES,
        choices=list(MODEL_SHORTNAMES.keys()),
        help="One or more model names supported by src.models.load_model",
    )
    parser.add_argument("--layers", nargs="+", type=int, help="Optional explicit list of layers to analyse")
    parser.add_argument(
        "--representation",
        default="final_token",
        choices=["final_token", "mean_pool"],
        help="Residual-stream summary to use for each prompt",
    )
    parser.add_argument(
        "--alpha_values",
        nargs="+",
        type=float,
        default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        help="Intervention strengths, measured in direction-projection standard deviations",
    )
    parser.add_argument("--results_dir", default="results/emotion_negation", help="Directory for CSVs and vector artifacts")
    parser.add_argument("--report_dir", default="reports", help="Directory for markdown reports")
    return parser.parse_args()


def _save_direction_artifacts(result: EmotionDirectionResult, output_path: str) -> str:
    payload: dict[str, np.ndarray] = {}
    for (emotion, layer), vector in result.direction_vectors.items():
        payload[f"direction__{emotion}__layer_{layer}"] = vector
    for (emotion, prompt_kind, layer), vector in result.mean_vectors.items():
        payload[f"mean__{emotion}__{prompt_kind}__layer_{layer}"] = vector

    np.savez_compressed(output_path, **payload)
    return output_path


def _format_table(df: pd.DataFrame, columns: list[str], float_cols: set[str] | None = None) -> str:
    if df.empty:
        return "No rows available.\n"

    float_cols = float_cols or set()
    subset = df.loc[:, columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in subset.itertuples(index=False, name=None):
        values = []
        for col, value in zip(columns, row):
            if pd.isna(value):
                values.append("")
            elif col in float_cols:
                values.append(f"{float(value):.4f}")
            elif isinstance(value, (np.integer, int)):
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _peak_summary(summary_df: pd.DataFrame, reference_layers: dict[str, int]) -> pd.DataFrame:
    peak_rows = summary_df.loc[summary_df.groupby("emotion")["direction_norm"].idxmax()].copy()
    peak_rows = peak_rows.rename(
        columns={
            "layer": "peak_layer",
            "direction_norm": "peak_direction_norm",
            "probe_accuracy": "peak_probe_accuracy",
        }
    )

    early_cutoff = reference_layers["middle"]
    early_means = (
        summary_df.assign(region=np.where(summary_df["layer"] < early_cutoff, "early", "late"))
        .pivot_table(index="emotion", columns="region", values="direction_norm", aggfunc="mean")
        .reset_index()
    )
    merged = peak_rows.merge(early_means, on="emotion", how="left")
    merged["late_over_early"] = merged["late"] / merged["early"]
    columns = [
        "emotion",
        "peak_layer",
        "peak_direction_norm",
        "peak_probe_accuracy",
        "early",
        "late",
        "late_over_early",
    ]
    return merged.loc[:, columns].sort_values("emotion").reset_index(drop=True)


def _peak_negation_summary(peak_df: pd.DataFrame, negation_df: pd.DataFrame) -> pd.DataFrame:
    merged = peak_df[["emotion", "peak_layer"]].merge(
        negation_df,
        left_on=["emotion", "peak_layer"],
        right_on=["emotion", "layer"],
        how="left",
    )
    merged["distance_gap"] = merged["distance_to_opposite"] - merged["distance_to_neutral"]
    return merged[
        [
            "emotion",
            "peak_layer",
            "opposite_emotion",
            "distance_to_neutral",
            "distance_to_opposite",
            "distance_gap",
            "closer_to_neutral_rate",
        ]
    ].sort_values("emotion").reset_index(drop=True)


def _peak_linearity_summary(peak_df: pd.DataFrame, linearity_summary: pd.DataFrame) -> pd.DataFrame:
    if linearity_summary.empty:
        return pd.DataFrame()
    work = linearity_summary[linearity_summary["base_condition"] == "affirmed_prefix"].copy()
    work = work.rename(columns={"layer": "peak_layer"})
    return peak_df[["emotion", "peak_layer"]].merge(work, on=["emotion", "peak_layer"], how="left").sort_values("emotion").reset_index(drop=True)


def _supportive_findings(peak_df: pd.DataFrame, negation_peak_df: pd.DataFrame, linearity_peak_df: pd.DataFrame, reference_layers: dict[str, int]) -> list[str]:
    findings: list[str] = []

    if not peak_df.empty and (peak_df["peak_layer"] >= reference_layers["middle"]).all():
        findings.append("All emotions peak in the middle-to-late layers, which supports the claim that negation-sensitive directions consolidate late.")

    strong_probe = peak_df.dropna(subset=["peak_probe_accuracy"])
    if not strong_probe.empty and (strong_probe["peak_probe_accuracy"] >= 0.9).all():
        findings.append("Peak-layer probe accuracy stays high across emotions, so affirmed and negated prompts are linearly separable once the representation is formed.")

    neutral_rows = negation_peak_df.dropna(subset=["closer_to_neutral_rate"])
    if not neutral_rows.empty:
        winners = neutral_rows[neutral_rows["closer_to_neutral_rate"] > 0.5]
        if not winners.empty:
            findings.append(
                "At least some emotions move closer to neutral than to their opposite under negation, which helps the attenuation-not-flip interpretation."
            )

    if not linearity_peak_df.empty:
        strong_linear = linearity_peak_df.dropna(subset=["linearity_r2"])
        strong_linear = strong_linear[strong_linear["linearity_r2"] >= 0.8]
        if not strong_linear.empty:
            findings.append("Some peak-layer direction injections remain reasonably linear, so the learned directions are not purely descriptive.")

    return findings


def _contradictory_findings(negation_peak_df: pd.DataFrame, linearity_peak_df: pd.DataFrame) -> list[str]:
    findings: list[str] = []

    neutral_rows = negation_peak_df.dropna(subset=["closer_to_neutral_rate"])
    if not neutral_rows.empty:
        weak_rows = neutral_rows[neutral_rows["closer_to_neutral_rate"] <= 0.5]
        if not weak_rows.empty:
            emotions = ", ".join(f"`{emotion}`" for emotion in weak_rows["emotion"].tolist())
            findings.append(f"{emotions} do not reliably move toward neutral at the peak layer, which weakens a universal attenuation claim.")

    if not linearity_peak_df.empty:
        weak_linear = linearity_peak_df.dropna(subset=["linearity_r2"])
        weak_linear = weak_linear[weak_linear["linearity_r2"] < 0.5]
        if not weak_linear.empty:
            emotions = ", ".join(f"`{emotion}`" for emotion in weak_linear["emotion"].tolist())
            findings.append(f"{emotions} show weak linearity at the peak layer, so the causal direction story is uneven across emotions.")

    missing_opposites = negation_peak_df["opposite_emotion"].isna().sum()
    if missing_opposites > 0:
        findings.append("Some emotions still lack explicit opposite-emotion controls, so the neutral-vs-opposite conclusion is incomplete.")

    return findings


def write_markdown_report(
    model_name: str,
    dataset_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    negation_peak_df: pd.DataFrame,
    linearity_peak_df: pd.DataFrame,
    report_path: str,
    representation: str,
    reference_layers: dict[str, int],
) -> str:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    support_lines = _supportive_findings(peak_df, negation_peak_df, linearity_peak_df, reference_layers)
    contradiction_lines = _contradictory_findings(negation_peak_df, linearity_peak_df)

    lines = [
        f"# Emotion Negation Report: {model_name}",
        "",
        "## Setup",
        "",
        f"- Model: `{model_name}`",
        f"- Representation: `{representation}`",
        f"- Prompt examples: `{len(dataset_df)}`",
        f"- Emotion pairs: `{int((dataset_df['prompt_kind'] == 'affirmed').sum())}` affirmed and `{int((dataset_df['prompt_kind'] == 'negated').sum())}` negated",
        f"- Neutral controls: `{int((dataset_df['prompt_kind'] == 'neutral').sum())}`",
        f"- Reference layers: `{reference_layers}`",
        "",
        "## Peak Direction Summary",
        "",
        _format_table(
            peak_df,
            ["emotion", "peak_layer", "peak_direction_norm", "peak_probe_accuracy", "early", "late", "late_over_early"],
            float_cols={"peak_direction_norm", "peak_probe_accuracy", "early", "late", "late_over_early"},
        ),
        "Interpretation:",
        "",
        "- `peak_direction_norm` is the strongest affirmed-vs-negated separation for that emotion.",
        "- `late_over_early` compares mean late-layer strength to early-layer strength; values above 1 mean the direction gets stronger deeper in the network.",
        "",
        "## Negation Behaviour At The Peak Layer",
        "",
        _format_table(
            negation_peak_df,
            ["emotion", "peak_layer", "opposite_emotion", "distance_to_neutral", "distance_to_opposite", "distance_gap", "closer_to_neutral_rate"],
            float_cols={"distance_to_neutral", "distance_to_opposite", "distance_gap", "closer_to_neutral_rate"},
        ),
        "Interpretation:",
        "",
        "- Negative `distance_gap` means the negated representation is closer to the opposite emotion than to neutral.",
        "- `closer_to_neutral_rate` above 0.5 means negation behaves more like attenuation than a polarity flip for most prompts.",
        "",
    ]

    if not linearity_peak_df.empty:
        lines.extend(
            [
                "## Peak-Layer Direction Injection",
                "",
                _format_table(
                    linearity_peak_df,
                    ["emotion", "peak_layer", "contrast_label", "slope", "linearity_r2", "start_margin", "end_margin"],
                    float_cols={"slope", "linearity_r2", "start_margin", "end_margin"},
                ),
                "Interpretation:",
                "",
                "- Positive `slope` means moving along the learned direction increases target emotion mass relative to the contrast set.",
                "- Higher `linearity_r2` means the effect changes more predictably as the intervention strength grows.",
                "",
            ]
        )

    lines.extend(
        [
            "## What Helps The Story",
            "",
        ]
    )
    lines.extend([f"- {line}" for line in support_lines] or ["- No strong supporting pattern emerged in this rerun."])
    lines.extend(
        [
            "",
            "## What Hurts The Story",
            "",
        ]
    )
    lines.extend([f"- {line}" for line in contradiction_lines] or ["- No strong contradictory pattern emerged in this rerun."])
    lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            "- This report is text-only by design: no PCA, heatmaps, or line plots are used in the write-up.",
            "- The most useful pieces here are where the direction peaks, whether it strengthens late, and whether negation moves emotions toward neutral or somewhere else.",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return report_path


def run_emotion_negation_report(
    model_names: list[str] | None = None,
    layers: list[int] | None = None,
    representation: str = "final_token",
    alpha_values: list[float] | None = None,
    results_dir: str = "results/emotion_negation",
    report_dir: str = "reports",
) -> dict[str, dict[str, str]]:
    model_names = model_names or CANONICAL_MODEL_NAMES
    outputs: dict[str, dict[str, str]] = {}

    for raw_model_name in model_names:
        model_name = MODEL_SHORTNAMES.get(raw_model_name, raw_model_name)
        print("=" * 70)
        print(f"Emotion negation experiment: {model_name}")
        print("=" * 70)

        model = load_model(model_name)
        dataset = build_emotion_prompt_dataset(model=model, verbose=True)
        result = analyze_emotion_negation(
            model=model,
            dataset=dataset,
            layers=layers,
            representation=representation,
            alpha_values=alpha_values,
            verbose=True,
        )

        model_results_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        dataset_csv = os.path.join(model_results_dir, "emotion_prompt_metadata.csv")
        summary_csv = os.path.join(model_results_dir, "emotion_direction_summary.csv")
        negation_csv = os.path.join(model_results_dir, "emotion_negation_sensitivity.csv")
        linearity_summary_csv = os.path.join(model_results_dir, "emotion_linearity_summary.csv")
        vectors_npz = os.path.join(model_results_dir, "emotion_direction_vectors.npz")
        report_path = os.path.join(report_dir, f"emotion_negation_{model_name}.md")

        peak_df = _peak_summary(result.summary, result.reference_layers)
        negation_peak_df = _peak_negation_summary(peak_df, result.negation_sensitivity)
        linearity_peak_df = _peak_linearity_summary(peak_df, result.linearity_summary)

        result.metadata.to_csv(dataset_csv, index=False)
        peak_df.to_csv(summary_csv, index=False)
        negation_peak_df.to_csv(negation_csv, index=False)
        linearity_peak_df.to_csv(linearity_summary_csv, index=False)
        _save_direction_artifacts(result, vectors_npz)

        write_markdown_report(
            model_name=model_name,
            dataset_df=result.metadata,
            peak_df=peak_df,
            negation_peak_df=negation_peak_df,
            linearity_peak_df=linearity_peak_df,
            report_path=report_path,
            representation=representation,
            reference_layers=result.reference_layers,
        )

        print(f"Saved results to: {model_results_dir}")
        print(f"Saved report to:  {report_path}")

        outputs[model_name] = {
            "dataset_csv": dataset_csv,
            "summary_csv": summary_csv,
            "negation_csv": negation_csv,
            "linearity_summary_csv": linearity_summary_csv,
            "vectors_npz": vectors_npz,
            "report_path": report_path,
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outputs


def main():
    args = parse_args()
    run_emotion_negation_report(
        model_names=args.models,
        layers=args.layers,
        representation=args.representation,
        alpha_values=args.alpha_values,
        results_dir=args.results_dir,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
