from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _flatten_numeric(values: Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(float).reshape(-1)

    chunks = []
    for value in values:
        chunks.append(np.asarray(value, dtype=float).reshape(-1))
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks)


def dynamic_axis_limits(
    values: Iterable[float] | np.ndarray,
    floor: float | None = None,
    ceil: float | None = None,
    pad_ratio: float = 0.08,
) -> tuple[float, float]:
    vals = _flatten_numeric(values)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(vals.min())
        hi = float(vals.max())
        if np.isclose(lo, hi):
            pad = max(abs(lo) * pad_ratio, 0.1)
        else:
            pad = (hi - lo) * pad_ratio
        lo -= pad
        hi += pad

    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    if np.isclose(lo, hi):
        hi = lo + 1.0

    return lo, hi
