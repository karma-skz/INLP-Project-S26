"""
Summary helpers for the Signal-to-Gate Ratio (SGR) benchmark.

The original project generated several benchmark plots from these results.
This module now keeps the analysis text-first so the reports can stay
compact and easier to explain.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.notna().all():
        return mapped.astype(bool)
    return series.astype(bool)


def _finite_sgr(series: pd.Series) -> pd.Series:
    sgr = pd.to_numeric(series, errors="coerce")
    return sgr.replace([float("inf"), -float("inf")], np.nan)


def analyse_sgr_distribution(
    df: pd.DataFrame,
    fig_dir: str = "figures",
    sgr_clip: Optional[float] = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build text-ready SGR summaries.

    Parameters are kept backward-compatible with the older plotting helper,
    but ``fig_dir`` and ``sgr_clip`` are no longer used.
    """
    del fig_dir, sgr_clip

    work = df.copy()
    if "model_name" not in work.columns:
        work["model_name"] = "model"

    work["negation_failure"] = _coerce_bool(work["negation_failure"])
    work["rank_shift"] = work["neg_target_rank"] - work["pos_target_rank"]
    work["sgr_finite"] = _finite_sgr(work["sgr"])

    benchmark_rows: list[dict[str, float | int | str]] = []
    outcome_rows: list[dict[str, float | int | str]] = []
    edge_rows: list[dict[str, float | int | str]] = []

    for model_name, group in work.groupby("model_name"):
        failures = group[group["negation_failure"]]
        successes = group[~group["negation_failure"]]

        success_sgr = successes["sgr_finite"].dropna()
        failure_sgr = failures["sgr_finite"].dropna()
        finite_sgr = group["sgr_finite"].dropna()

        benchmark_rows.append(
            {
                "model_name": model_name,
                "n_samples": int(len(group)),
                "n_failures": int(group["negation_failure"].sum()),
                "failure_rate": float(group["negation_failure"].mean()),
                "median_rank_shift": float(group["rank_shift"].median()),
                "mean_rank_shift": float(group["rank_shift"].mean()),
                "median_sgr": float(finite_sgr.median()) if not finite_sgr.empty else np.nan,
                "mean_sgr": float(finite_sgr.mean()) if not finite_sgr.empty else np.nan,
                "success_median_sgr": float(success_sgr.median()) if not success_sgr.empty else np.nan,
                "failure_median_sgr": float(failure_sgr.median()) if not failure_sgr.empty else np.nan,
                "sgr_gt1_rate": float((finite_sgr > 1).mean()) if not finite_sgr.empty else np.nan,
            }
        )

        for outcome_name, outcome_group in (("success", successes), ("failure", failures)):
            outcome_sgr = outcome_group["sgr_finite"].dropna()
            outcome_rows.append(
                {
                    "model_name": model_name,
                    "outcome": outcome_name,
                    "n_samples": int(len(outcome_group)),
                    "median_rank_shift": float(outcome_group["rank_shift"].median()) if len(outcome_group) else np.nan,
                    "mean_rank_shift": float(outcome_group["rank_shift"].mean()) if len(outcome_group) else np.nan,
                    "median_sgr": float(outcome_sgr.median()) if not outcome_sgr.empty else np.nan,
                    "mean_sgr": float(outcome_sgr.mean()) if not outcome_sgr.empty else np.nan,
                }
            )

        success_with_high_sgr = int((successes["sgr_finite"] > 1).sum())
        failure_with_low_sgr = int((failures["sgr_finite"] <= 1).sum())
        edge_rows.append(
            {
                "model_name": model_name,
                "success_with_sgr_gt1": success_with_high_sgr,
                "failure_with_sgr_le1": failure_with_low_sgr,
                "success_mismatch_rate": (
                    success_with_high_sgr / len(successes) if len(successes) else np.nan
                ),
                "failure_mismatch_rate": (
                    failure_with_low_sgr / len(failures) if len(failures) else np.nan
                ),
            }
        )

    benchmark_df = pd.DataFrame(benchmark_rows)
    outcome_df = pd.DataFrame(outcome_rows)
    edge_df = pd.DataFrame(edge_rows)

    if verbose:
        print("\n" + "=" * 60)
        print("SGR Summary")
        print("=" * 60)
        for _, row in benchmark_df.iterrows():
            print(
                f"{row['model_name']:<12} "
                f"failure_rate={row['failure_rate']:.1%} "
                f"median_rank_shift={row['median_rank_shift']:.1f} "
                f"median_sgr={row['median_sgr']:.3f}"
            )

    return {
        "benchmark_summary": benchmark_df,
        "outcome_summary": outcome_df,
        "edge_cases": edge_df,
    }
