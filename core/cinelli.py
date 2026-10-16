"""Cinelli-Hazlett robustness values for a matched contrast (SI-2).

The regression stacks each pair into two rows and fits

    Y = a + τ * Treated + covariates + error.

Treatment marks the focal arm: AIGC-adopting, platform-labeled AIGC, or
population-feedback. Partial R^2 of treatment and the robustness value
RV_{q=1} follow Cinelli and Hazlett (2020). RV at α = 0.05 is the value
that would bring the estimate below significance at 5%. The benchmark is
the observed covariate with the largest partial R^2 for treatment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from core.columns import COMPARISON, FOCAL


def robustness_table(pairs, outcomes, numeric_cols):
    rows = []
    for outcome in outcomes:
        rows.append(_one_outcome(pairs, outcome, numeric_cols))
    return pd.DataFrame(rows)


def _one_outcome(pairs, outcome, numeric_cols):
    focal_rows = _side(pairs, outcome, numeric_cols, suffix=FOCAL, arm=1)
    comparison_rows = _side(pairs, outcome, numeric_cols, suffix=COMPARISON, arm=0)
    frame = pd.concat([focal_rows, comparison_rows], ignore_index=True).dropna()
    outcome_values = frame["outcome_value"].to_numpy(float)
    varying = [column for column in numeric_cols if frame[column].std(ddof=0) > 0]
    design = sm.add_constant(frame[["treated"] + varying].astype(float), has_constant="add")
    fit = sm.OLS(outcome_values, design).fit()
    t_statistic = float(fit.tvalues["treated"])
    degrees_of_freedom = float(fit.df_resid)
    partial_r2 = _partial_r2_from_t(t_statistic, degrees_of_freedom)
    benchmark, treatment_r2, outcome_r2 = _strongest_covariate(frame, varying)
    return {
        "outcome": outcome,
        "estimate": float(fit.params["treated"]),
        "t_statistic": t_statistic,
        "partial_r2_outcome_on_treatment": partial_r2,
        "robustness_value": _robustness_value(partial_r2),
        "robustness_value_alpha_05": _robustness_value_at_alpha(t_statistic, degrees_of_freedom, alpha=0.05),
        "benchmark_covariate": benchmark,
        "partial_r2_treatment_on_covariate": treatment_r2,
        "partial_r2_outcome_on_covariate": outcome_r2,
    }


def _side(pairs, outcome, numeric_cols, suffix, arm):
    data = {column: pairs[f"{column}{suffix}"] for column in numeric_cols}
    data["outcome_value"] = pairs[f"{outcome}{suffix}"]
    data["treated"] = arm
    return pd.DataFrame(data)


def _partial_r2_from_t(t_stat, dof):
    t2 = t_stat ** 2
    return float(t2 / (t2 + dof))


def _robustness_value(partial_r2):
    if not np.isfinite(partial_r2) or partial_r2 >= 1:
        return np.nan
    strength = partial_r2 / (1 - partial_r2)
    return float(0.5 * (np.sqrt(strength**2 + 4 * strength) - strength))


def _robustness_value_at_alpha(t_stat, dof, alpha):
    critical = stats.t.ppf(1 - alpha / 2, dof)
    reduced = abs(t_stat) / np.sqrt(dof) - critical / np.sqrt(dof)
    if reduced <= 0:
        return 0.0
    strength = reduced ** 2
    return float(0.5 * (np.sqrt(strength**2 + 4 * strength) - strength))


def _strongest_covariate(frame, columns):
    strongest_name = ""
    strongest_treatment_r2 = np.nan
    strongest_outcome_r2 = np.nan
    if not columns:
        return strongest_name, strongest_treatment_r2, strongest_outcome_r2
    for column in columns:
        other_columns = [name for name in columns if name != column]
        treatment_r2 = _nested_partial_r2(
            frame["treated"], frame[column], frame[other_columns] if other_columns else None
        )
        controls = frame[other_columns + ["treated"]] if other_columns else frame[["treated"]]
        outcome_r2 = _nested_partial_r2(frame["outcome_value"], frame[column], controls)
        if not np.isfinite(strongest_treatment_r2) or treatment_r2 > strongest_treatment_r2:
            strongest_name = column
            strongest_treatment_r2 = treatment_r2
            strongest_outcome_r2 = outcome_r2
    return strongest_name, float(strongest_treatment_r2), float(strongest_outcome_r2)


def _nested_partial_r2(y, x, controls):
    y = np.asarray(y, float)
    pieces = [np.ones(len(y)), np.asarray(x, float)]
    if controls is not None and len(controls.columns):
        pieces.append(np.asarray(controls, float))
    full = np.column_stack(pieces)
    reduced = full[:, [0] + list(range(2, full.shape[1]))] if full.shape[1] > 2 else full[:, [0]]
    r2_full = sm.OLS(y, full).fit().rsquared
    r2_reduced = sm.OLS(y, reduced).fit().rsquared
    denom = 1 - r2_reduced
    if denom <= 1e-12:
        return np.nan
    return float((r2_full - r2_reduced) / denom)
