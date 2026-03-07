"""
src/metrics/metrics.py
========================
Statistical analysis of the benchmark results.

Functions
---------
  negation_failure_rate   — Overall / per-model failure rate with 95% CI
  sgr_vs_failure_correlation — Spearman / point-biserial correlation of SGR
                               with negation failure flag
  bootstrap_ci            — Non-parametric bootstrap confidence interval
  summary_stats           — Full report as a dict (printed as a table)

Usage
-----
    import pandas as pd
    from src.metrics import summary_stats, sgr_vs_failure_correlation

    df = pd.read_csv("results/gpt2_benchmark.csv")
    print(summary_stats(df))
    print(sgr_vs_failure_correlation(df))
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_sgr(df: pd.DataFrame) -> pd.Series:
    """Return SGR column with inf replaced by NaN (for numeric operations)."""
    sgr = df["sgr"].copy().astype(float)
    sgr.replace([float("inf"), -float("inf")], np.nan, inplace=True)
    return sgr


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    values : 1-D array-like
    statistic : callable  (default: np.mean)
    n_boot : int
    ci : float  (e.g. 0.95 for 95%)
    rng : optional numpy Generator

    Returns
    -------
    (point_estimate, lower_ci, upper_ci)
    """
    rng   = rng or np.random.default_rng(42)
    vals  = np.asarray(values, dtype=float)
    vals  = vals[~np.isnan(vals)]
    point = statistic(vals)
    boot_stats = [
        statistic(rng.choice(vals, size=len(vals), replace=True))
        for _ in range(n_boot)
    ]
    alpha  = (1 - ci) / 2
    lower  = np.quantile(boot_stats, alpha)
    upper  = np.quantile(boot_stats, 1 - alpha)
    return float(point), float(lower), float(upper)


# ---------------------------------------------------------------------------
# Failure rate
# ---------------------------------------------------------------------------

def negation_failure_rate(
    df: pd.DataFrame,
    ci: float = 0.95,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """
    Compute negation failure rate per model, with bootstrap CI.

    Returns a DataFrame with columns:
      model_name, n_samples, n_failures, failure_rate, ci_lower, ci_upper
    """
    if "model_name" not in df.columns:
        df = df.copy()
        df["model_name"] = "model"

    rows = []
    for model, grp in df.groupby("model_name"):
        y = grp["negation_failure"].astype(float).values
        point, lo, hi = bootstrap_ci(y, statistic=np.mean, ci=ci, n_boot=n_boot)
        rows.append({
            "model_name":   model,
            "n_samples":    len(grp),
            "n_failures":   int(grp["negation_failure"].sum()),
            "failure_rate": point,
            "ci_lower":     lo,
            "ci_upper":     hi,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SGR ↔ failure correlation
# ---------------------------------------------------------------------------

def sgr_vs_failure_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute three correlation statistics between SGR and the binary negation
    failure flag, per model:

    1. Spearman rank correlation (SGR, failure)
    2. Point-biserial correlation (SGR, failure)
    3. Mann-Whitney U  — SGR distribution of failures vs successes

    Returns a DataFrame with correlation statistics per model.
    """
    if "model_name" not in df.columns:
        df = df.copy()
        df["model_name"] = "model"

    rows = []
    for model, grp in df.groupby("model_name"):
        sgr  = _finite_sgr(grp).values
        fail = grp["negation_failure"].astype(float).values

        # Drop NaN SGR rows
        mask = ~np.isnan(sgr)
        sgr_c, fail_c = sgr[mask], fail[mask]

        # Spearman
        spearman_r, spearman_p = stats.spearmanr(sgr_c, fail_c)

        # Point-biserial
        pb_r, pb_p = stats.pointbiserialr(fail_c, sgr_c)

        # Mann-Whitney U (SGR in failure group vs success group)
        sgr_fail    = sgr_c[fail_c == 1]
        sgr_success = sgr_c[fail_c == 0]
        if len(sgr_fail) > 0 and len(sgr_success) > 0:
            u_stat, u_p = stats.mannwhitneyu(
                sgr_fail, sgr_success, alternative="greater"
            )
            # Effect size r = Z / sqrt(N)
            n_total = len(sgr_fail) + len(sgr_success)
            z_score = stats.norm.isf(u_p / 2)
            r_effect = z_score / np.sqrt(n_total)
        else:
            u_stat, u_p, r_effect = np.nan, np.nan, np.nan

        rows.append({
            "model_name":         model,
            "n_samples":          len(grp),
            "spearman_r":         spearman_r,
            "spearman_p":         spearman_p,
            "pointbiserial_r":    pb_r,
            "pointbiserial_p":    pb_p,
            "mannwhitney_u":      u_stat,
            "mannwhitney_p":      u_p,
            "mannwhitney_effect": r_effect,
            "mean_sgr_failure":   float(np.nanmean(sgr_fail))   if len(sgr_fail)    > 0 else np.nan,
            "mean_sgr_success":   float(np.nanmean(sgr_success)) if len(sgr_success) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full summary
# ---------------------------------------------------------------------------

def summary_stats(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Dict:
    """
    Compute and (optionally) print a comprehensive summary of results.

    Returns a nested dict with keys per model:
      failure_rate, sgr_mean, sgr_median, sgr_std,
      correlation_spearman, correlation_pb,
      crossover_present_rate (fraction of samples that have a crossover layer)
    """
    if "model_name" not in df.columns:
        df = df.copy()
        df["model_name"] = "model"

    out: Dict = {}

    for model, grp in df.groupby("model_name"):
        sgr_s  = _finite_sgr(grp)
        n      = len(grp)
        nf     = int(grp["negation_failure"].sum())

        spearman_r, spearman_p = stats.spearmanr(
            sgr_s.dropna().values,
            grp.loc[sgr_s.notna(), "negation_failure"].astype(float).values
        )

        crossover_rate = grp["crossover_layer"].notna().mean() if "crossover_layer" in grp.columns else np.nan

        out[model] = {
            "n_samples":          n,
            "failure_rate":       nf / n,
            "sgr_mean":           float(sgr_s.mean()),
            "sgr_median":         float(sgr_s.median()),
            "sgr_std":            float(sgr_s.std()),
            "sgr_gt1_rate":       float((sgr_s > 1).mean()),
            "spearman_r":         float(spearman_r),
            "spearman_p":         float(spearman_p),
            "crossover_present":  float(crossover_rate),
        }

        if verbose:
            d = out[model]
            print(f"\n{'='*60}")
            print(f"Model: {model}   (n={n})")
            print(f"{'='*60}")
            print(f"  Negation failure rate          : {d['failure_rate']:.1%}   ({nf}/{n})")
            print(f"  SGR — mean / median / std      : "
                  f"{d['sgr_mean']:.3f} / {d['sgr_median']:.3f} / {d['sgr_std']:.3f}")
            print(f"  SGR > 1 rate                   : {d['sgr_gt1_rate']:.1%}")
            print(f"  Spearman SGR↔failure           : r={d['spearman_r']:.3f}  p={d['spearman_p']:.4f}")
            print(f"  Samples with crossover layer   : {d['crossover_present']:.1%}")

    return out


# ---------------------------------------------------------------------------
# Model comparison significance test
# ---------------------------------------------------------------------------

def compare_models(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> Dict:
    """
    Test whether the negation failure rates of two models are significantly
    different (two-proportion z-test) and whether their SGR distributions
    differ (Mann-Whitney U).

    Returns dict with test statistics and p-values.
    """
    sub_a = df[df["model_name"] == model_a]
    sub_b = df[df["model_name"] == model_b]

    if len(sub_a) == 0 or len(sub_b) == 0:
        raise ValueError(f"Must have data for both models: {model_a}, {model_b}")

    # Two-proportion z-test for failure rates
    n_a, nf_a = len(sub_a), int(sub_a["negation_failure"].sum())
    n_b, nf_b = len(sub_b), int(sub_b["negation_failure"].sum())
    p_pool    = (nf_a + nf_b) / (n_a + n_b)
    se        = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z         = ((nf_a / n_a) - (nf_b / n_b)) / se if se > 0 else np.nan
    p_z       = float(2 * stats.norm.sf(abs(z)))

    # Mann-Whitney on SGR
    sgr_a = _finite_sgr(sub_a).dropna().values
    sgr_b = _finite_sgr(sub_b).dropna().values
    u_stat, u_p = stats.mannwhitneyu(sgr_a, sgr_b, alternative="two-sided")

    result = {
        f"failure_rate_{model_a}": nf_a / n_a,
        f"failure_rate_{model_b}": nf_b / n_b,
        "two_proportion_z":        float(z),
        "two_proportion_p":        float(p_z),
        f"mean_sgr_{model_a}":     float(np.nanmean(sgr_a)),
        f"mean_sgr_{model_b}":     float(np.nanmean(sgr_b)),
        "mannwhitney_sgr_u":       float(u_stat),
        "mannwhitney_sgr_p":       float(u_p),
    }

    print(f"\nModel comparison: {model_a} vs {model_b}")
    print(f"  Failure rates    : {result[f'failure_rate_{model_a}']:.1%} vs {result[f'failure_rate_{model_b}']:.1%}")
    print(f"  Two-prop z-test  : z={z:.3f}  p={p_z:.4f}")
    print(f"  SGR Mann-Whitney : U={u_stat:.0f}  p={u_p:.4f}")

    return result
