"""Balance and overlap summaries for a matched comparison.

Standardized mean differences use the pooled standard deviation.
The variance ratio divides the focal-arm variance by the comparison-arm variance, using the manuscript orientation (adopting, platform-labeled AIGC, or population-feedback first).
The manuscript treats a covariate as balanced when the absolute standardized
difference is below 0.1 and the variance ratio is between 0.5 and 2.
Propensity scores are fit on the numeric covariates only, then the shared
histogram area on [0, 1] is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.columns import COMPARISON, FOCAL


def standardized_mean_differences(focal, comparison, columns):
    rows = []
    for column in columns:
        focal_values = pd.to_numeric(focal[column], errors="coerce")
        comparison_values = pd.to_numeric(comparison[column], errors="coerce")
        if focal_values.notna().sum() == 0 or comparison_values.notna().sum() == 0:
            continue
        focal_sd = focal_values.std(ddof=1)
        comparison_sd = comparison_values.std(ddof=1)
        pooled_sd = np.sqrt((focal_sd**2 + comparison_sd**2) / 2)
        if pooled_sd < 1e-8:
            difference = 0.0
        else:
            difference = float((focal_values.mean() - comparison_values.mean()) / pooled_sd)
        focal_var = float(focal_values.var(ddof=1))
        comparison_var = float(comparison_values.var(ddof=1))
        variance_ratio = focal_var / comparison_var if comparison_var > 1e-12 else np.nan
        rows.append(
            {
                "covariate": column,
                "mean_focal": float(focal_values.mean()),
                "mean_comparison": float(comparison_values.mean()),
                "standardized_difference": difference,
                "variance_ratio": variance_ratio,
            }
        )
    return pd.DataFrame(rows)


def balance_before_after(focal, comparison, pairs, columns):
    before = standardized_mean_differences(focal, comparison, columns).rename(
        columns={
            "standardized_difference": "standardized_difference_before",
            "mean_focal": "mean_focal_before",
            "mean_comparison": "mean_comparison_before",
            "variance_ratio": "variance_ratio_before",
        }
    )
    matched_focal = pd.DataFrame({column: pairs[f"{column}{FOCAL}"] for column in columns})
    matched_comparison = pd.DataFrame({column: pairs[f"{column}{COMPARISON}"] for column in columns})
    after = standardized_mean_differences(matched_focal, matched_comparison, columns).rename(
        columns={
            "standardized_difference": "standardized_difference_after",
            "mean_focal": "mean_focal_after",
            "mean_comparison": "mean_comparison_after",
            "variance_ratio": "variance_ratio_after",
        }
    )
    return before.merge(after, on="covariate", how="outer")


def propensity_overlap(focal, comparison, numeric_cols, n_bins=20):
    """Shared area of the two propensity histograms. NaN if the logit fails."""
    if not numeric_cols or len(focal) < 5 or len(comparison) < 5:
        return np.nan
    arm = np.concatenate([np.ones(len(focal)), np.zeros(len(comparison))])
    design = pd.concat([focal[numeric_cols], comparison[numeric_cols]], ignore_index=True)
    design = sm.add_constant(design.astype(float), has_constant="add")
    keep = [column for column in design.columns if column == "const" or design[column].std(ddof=0) > 0]
    design = design[keep]
    try:
        fitted = sm.Logit(arm, design).fit(disp=False, maxiter=100)
    except Exception:
        return np.nan
    scores = np.asarray(fitted.predict(), float)
    bins = np.linspace(0, 1, n_bins + 1)
    focal_hist, _ = np.histogram(scores[: len(focal)], bins=bins, density=True)
    comparison_hist, _ = np.histogram(scores[len(focal) :], bins=bins, density=True)
    width = bins[1] - bins[0]
    return float(np.minimum(focal_hist, comparison_hist).sum() * width)
