"""Differenced Granger test with a moving-block bootstrap p-value.

Methods, Granger causality (Fig. 2(b), left):

    ΔM_t = α + Σ_i β_i ΔM_{t-i} + Σ_i γ_i Δs^a_{t-i} + Σ_i δ_i Δs^u_{t-i} + λ D_t + ε_t

M_t is the first-day median exposure of platform-labeled AIGC videos uploaded on day t.
s_t^a and s_t^u are the upload counts of platform-labeled AIGC and unlabeled content.
D_t holds weekday, holiday, campaign-period, and month indicators, plus the
log of vertical-wide exposure. The null is γ_1 = … = γ_p = 0.
The p-value is a residual moving-block bootstrap under that restricted
model: residuals from the regression without the γ lags are resampled
in blocks, added back to the restricted fitted values, and the unrestricted
F statistic is recomputed. Blocks are 7 days.
Coefficient intervals use Newey-West standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.exog import control_matrix


def granger_test(
    frame,
    outcome,
    focal,
    comparison,
    exog_cols,
    lag,
    hac_lags,
    n_bootstrap,
    block_length,
    seed,
):
    design = _design(frame, outcome, focal, comparison, exog_cols, lag)
    hac_fit, ols_fit, focal_terms = _fit(design, hac_lags)
    observed_f = _f_statistic(ols_fit, focal_terms)
    bootstrap_p, draws_used = _bootstrap(design, focal_terms, observed_f, n_bootstrap, block_length, seed)
    rows = []
    for term in focal_terms:
        rows.append(
            {
                "term": term,
                "estimate": float(hac_fit.params[term]),
                "standard_error": float(hac_fit.bse[term]),
                "lag": lag,
                "f_statistic": observed_f,
                "bootstrap_p": bootstrap_p,
                "bootstrap_draws": draws_used,
            }
        )
    return pd.DataFrame(rows)


def _design(frame, outcome, focal, comparison, exog_cols, lag):
    changes = pd.DataFrame(
        {
            "outcome_change": frame[outcome].diff(),
            "focal_change": frame[focal].diff(),
            "comparison_change": frame[comparison].diff(),
        }
    )
    for lag_index in range(1, lag + 1):
        changes[f"outcome_lag_{lag_index}"] = changes["outcome_change"].shift(lag_index)
        changes[f"focal_lag_{lag_index}"] = changes["focal_change"].shift(lag_index)
        changes[f"comparison_lag_{lag_index}"] = changes["comparison_change"].shift(lag_index)
    controls = control_matrix(frame, exog_cols).reset_index(drop=True)
    changes = changes.reset_index(drop=True)
    return pd.concat([changes, controls], axis=1).dropna().reset_index(drop=True)


def _fit(design, hac_lags):
    focal_terms = [column for column in design.columns if column.startswith("focal_lag_")]
    outcome_change = design["outcome_change"]
    regressors = design.drop(columns=["outcome_change", "focal_change", "comparison_change"])
    ols_fit = sm.OLS(outcome_change, regressors).fit()
    hac_fit = sm.OLS(outcome_change, regressors).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    return hac_fit, ols_fit, focal_terms


def _f_statistic(fit, terms):
    hypothesis = ", ".join(f"{term} = 0" for term in terms)
    return float(np.asarray(fit.f_test(hypothesis).fvalue).squeeze())


def _bootstrap(design, focal_terms, observed_f, n_bootstrap, block_length, seed):
    """Residual block bootstrap with the focal lags left out of the null fit."""
    rng = np.random.default_rng(seed)
    outcome_change = design["outcome_change"].to_numpy(float)
    regressors = design.drop(columns=["outcome_change", "focal_change", "comparison_change"])
    restricted_columns = [column for column in regressors.columns if column not in focal_terms]
    restricted = sm.OLS(outcome_change, regressors[restricted_columns]).fit()
    fitted = np.asarray(restricted.fittedvalues, float)
    residuals = np.asarray(restricted.resid, float)
    n_rows = len(design)
    block_length = max(1, min(int(block_length), n_rows // 2))
    starts = np.arange(0, n_rows - block_length + 1)
    n_blocks = int(np.ceil(n_rows / block_length))
    n_exceeding = 0
    draws_used = 0
    for _ in range(int(n_bootstrap)):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        draw_index = np.concatenate([np.arange(start, start + block_length) for start in chosen])[:n_rows]
        outcome_draw = fitted + residuals[draw_index]
        try:
            draw_fit = sm.OLS(outcome_draw, regressors).fit()
            draw_f = _f_statistic(draw_fit, focal_terms)
        except Exception:
            continue
        if not np.isfinite(draw_f):
            continue
        draws_used += 1
        n_exceeding += int(draw_f >= observed_f)
    bootstrap_p = (1 + n_exceeding) / (1 + draws_used) if draws_used else np.nan
    return float(bootstrap_p), draws_used

def granger_design(frame, outcome, focal, comparison, exog_cols, lag):
    """Differenced lags and controls that enter the Granger equation."""
    return _design(frame, outcome, focal, comparison, exog_cols, lag)
