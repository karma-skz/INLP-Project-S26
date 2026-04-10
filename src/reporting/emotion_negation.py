from __future__ import annotations

import argparse
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.analysis import EmotionDirectionResult, analyze_emotion_negation
from src.dataset import build_emotion_prompt_dataset
from src.models import CANONICAL_MODEL_NAMES, MODEL_SHORTNAMES, load_model
from src.utils import dynamic_axis_limits


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
    parser.add_argument("--results_dir", default="results/emotion_negation", help="Directory for metric CSV artifacts")
    parser.add_argument("--figures_dir", default="figures/emotion_negation", help="Directory for easy-to-read metric plots")
    return parser.parse_args()


def _save_metric_plots(
    model_name: str,
    peak_df: pd.DataFrame,
    negation_peak_df: pd.DataFrame,
    linearity_peak_df: pd.DataFrame,
    output_dir: str,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    if not peak_df.empty:
        plot_df = peak_df.sort_values("emotion").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(plot_df["emotion"], plot_df["peak_direction_norm"], color="#4C78A8")
        for bar, layer in zip(bars, plot_df["peak_layer"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"L{int(layer)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        lo, hi = dynamic_axis_limits(plot_df["peak_direction_norm"], floor=0.0)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{model_name}: Peak direction strength by emotion")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Peak direction norm")
        fig.tight_layout()
        out = os.path.join(output_dir, "peak_direction_strength.png")
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)

    if not negation_peak_df.empty:
        gap_df = negation_peak_df.sort_values("emotion").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(gap_df["emotion"], gap_df["distance_gap"], color="#F58518")
        ax.axhline(0.0, color="black", linewidth=1)
        lo, hi = dynamic_axis_limits(gap_df["distance_gap"])
        ax.set_ylim(lo, hi)
        ax.set_title(f"{model_name}: Negation distance gap (opposite - neutral)")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Distance gap")
        fig.tight_layout()
        out = os.path.join(output_dir, "negation_distance_gap.png")
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)

        rate_df = negation_peak_df.dropna(subset=["closer_to_neutral_rate"]).sort_values("emotion").reset_index(drop=True)
        if not rate_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(rate_df["emotion"], rate_df["closer_to_neutral_rate"], color="#54A24B")
            ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{model_name}: Negation moves toward neutral")
            ax.set_xlabel("Emotion")
            ax.set_ylabel("Closer-to-neutral rate")
            fig.tight_layout()
            out = os.path.join(output_dir, "closer_to_neutral_rate.png")
            fig.savefig(out, dpi=180)
            plt.close(fig)
            paths.append(out)

    if not linearity_peak_df.empty and {"emotion", "slope", "linearity_r2"}.issubset(linearity_peak_df.columns):
        lin_df = linearity_peak_df.sort_values("emotion").reset_index(drop=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].bar(lin_df["emotion"], lin_df["slope"], color="#B279A2")
        axes[0].axhline(0.0, color="black", linewidth=1)
        lo, hi = dynamic_axis_limits(lin_df["slope"])
        axes[0].set_ylim(lo, hi)
        axes[0].set_title("Injection slope")
        axes[0].set_xlabel("Emotion")
        axes[0].set_ylabel("Slope")

        axes[1].bar(lin_df["emotion"], lin_df["linearity_r2"], color="#72B7B2")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Linearity ($R^2$)")
        axes[1].set_xlabel("Emotion")
        axes[1].set_ylabel("$R^2$")

        fig.suptitle(f"{model_name}: Peak-layer intervention metrics")
        fig.tight_layout()
        out = os.path.join(output_dir, "intervention_linearity.png")
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)

    return paths


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


def run_emotion_negation_report(
    model_names: list[str] | None = None,
    layers: list[int] | None = None,
    representation: str = "final_token",
    alpha_values: list[float] | None = None,
    results_dir: str = "results/emotion_negation",
    figures_dir: str = "figures/emotion_negation",
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
        model_figures_dir = os.path.join(figures_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        summary_csv = os.path.join(model_results_dir, "emotion_direction_summary.csv")
        negation_csv = os.path.join(model_results_dir, "emotion_negation_sensitivity.csv")
        linearity_summary_csv = os.path.join(model_results_dir, "emotion_linearity_summary.csv")

        peak_df = _peak_summary(result.summary, result.reference_layers)
        negation_peak_df = _peak_negation_summary(peak_df, result.negation_sensitivity)
        linearity_peak_df = _peak_linearity_summary(peak_df, result.linearity_summary)

        peak_df.to_csv(summary_csv, index=False)
        negation_peak_df.to_csv(negation_csv, index=False)
        linearity_peak_df.to_csv(linearity_summary_csv, index=False)
        figure_paths = _save_metric_plots(
            model_name=model_name,
            peak_df=peak_df,
            negation_peak_df=negation_peak_df,
            linearity_peak_df=linearity_peak_df,
            output_dir=model_figures_dir,
        )

        print(f"Saved results to: {model_results_dir}")
        if figure_paths:
            print(f"Saved figures to: {model_figures_dir}")

        outputs[model_name] = {
            "summary_csv": summary_csv,
            "negation_csv": negation_csv,
            "linearity_summary_csv": linearity_summary_csv,
            "figures_dir": model_figures_dir,
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
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
