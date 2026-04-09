from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def safe_suffix(value: str) -> str:
    cleaned = value.strip().replace(" ", "-")
    return cleaned or "empty"


def benchmark_csv_path(results_dir: str | Path, model_name: str, negator_suffix: str = " not") -> Path:
    return Path(results_dir) / f"{model_name}_{safe_suffix(negator_suffix)}_benchmark.csv"


def resolve_benchmark_csv(results_dir: str | Path, model_name: str, negator_suffix: str = " not") -> Path:
    preferred = benchmark_csv_path(results_dir, model_name, negator_suffix)
    legacy = Path(results_dir) / f"{model_name}_benchmark.csv"

    for candidate in (preferred, legacy):
        if candidate.exists():
            return candidate

    return preferred


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.notna().all():
        return mapped.astype(bool)
    return series.astype(bool)


def load_benchmark_dataframe(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    df = pd.read_csv(path)

    if "negation_failure" in df.columns:
        df["negation_failure"] = _coerce_bool_series(df["negation_failure"])

    if "sgr" in df.columns:
        df["sgr"] = pd.to_numeric(df["sgr"], errors="coerce")
        df["sgr"] = df["sgr"].replace([np.inf, -np.inf], np.nan)

    return df
