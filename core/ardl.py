"""ARDL error-correction fit and Pesaran-Shin-Smith bounds F test.

The manuscript reparameterizes ARDL(p, q) as

    ΔY_t = c_0 + ρ Y_{t-1} + π' X_{t-1} + short-run differences + λ D_t + ξ_t

and reports the long-run multiplier θ = −π / ρ.
Standard errors for θ use the delta method and the Newey-West
covariance. The bounds F tests ρ = 0 and π = 0 jointly.
The manuscript's six equations use Case 3 (unrestricted intercept, no trend)
with k = 2 supply regressors. A level relationship is supported when
F ≥ 4.81 and the Newey-West t on ρ is at most −3.53. F below 3.80 is
unsupported. The band [3.80, 4.81) is inconclusive, and t does not resolve it.

Two manuscript uses, both estimated here:

- Fig. 2(b), right: Y_t = w_{k,t}, the 31-day exposure-tier share, with X_t = [s_t^a, s_t^u].
- Fig. 2(d): Y_t = ln(e_t), overall full-view or valid-view rate, with X_t = [ln(s_t^a), ln(s_t^u)].

Fig. 2(d) is the long-run elasticity of engagement depth. It is not a creator-supply elasticity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, breaks_cusumolsresid

from core.exog import control_matrix

# Case 3, 5% F bounds. k = 2 is the manuscript pair, 3.80 and 4.81.
BOUNDS_5 = {
    1: (4.94, 5.73),
    2: (3.80, 4.81),
    3: (3.23, 4.35),
}

# Case 3 t bound on ρ. The manuscript uses −3.53 at k = 2.
T_BOUND_5 = {1: -3.22, 2: -3.53, 3: -3.78, 4: -3.99}


def ardl_equation(frame, outcome, regressors, exog_cols, p, q, hac_lags):
    work = _ecm_frame(frame, outcome, regressors, exog_cols, p, q)
    y = work["dy"]
    x = work.drop(columns=["dy"])
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    classical = sm.OLS(y, x).fit()
    level_terms = ["y_l1"] + [f"{name}_l1" for name in regressors]
    bounds_f = float(np.asarray(fit.f_test(", ".join(f"{term} = 0" for term in level_terms)).fvalue).squeeze())
    k = len(regressors)
    lower, upper = BOUNDS_5.get(k, (np.nan, np.nan))
    try:
        breusch_godfrey_lag1 = float(acorr_breusch_godfrey(classical, nlags=1)[1])
    except Exception:
        breusch_godfrey_lag1 = np.nan
    try:
        breusch_godfrey_lag7 = float(acorr_breusch_godfrey(classical, nlags=min(7, max(len(y) // 5, 1)))[1])
    except Exception:
        breusch_godfrey_lag7 = np.nan
    stability = _recursive_stability(np.asarray(y, float), np.asarray(classical.model.exog, float))
    try:
        cusum_p = float(breaks_cusumolsresid(classical.resid)[1])
    except Exception:
        cusum_p = np.nan
    rho = float(fit.params["y_l1"])
    t_rho = float(fit.tvalues["y_l1"])
    t_bound = T_BOUND_5.get(k, np.nan)
    if not np.isfinite(lower):
        bounds = "not tabulated"
    elif bounds_f < lower:
        bounds = "unsupported"
    elif bounds_f < upper:
        bounds = "inconclusive"
    elif np.isfinite(t_bound) and t_rho <= t_bound:
        bounds = "supported"
    else:
        bounds = "criteria do not establish a level relationship"
    rows = []
    cov = fit.cov_params()
    for name in regressors:
        pi_name = f"{name}_l1"
        pi = float(fit.params[pi_name])
        theta, standard_error = _theta(rho, pi, cov, "y_l1", pi_name)
        rows.append(
            {
                "outcome": outcome,
                "regressor": name,
                "p": int(p),
                "q": int(q),
                "rho": rho,
                "pi": pi,
                "theta": theta,
                "theta_standard_error": standard_error,
                "theta_ci95_low": theta - 1.96 * standard_error if np.isfinite(standard_error) else np.nan,
                "theta_ci95_high": theta + 1.96 * standard_error if np.isfinite(standard_error) else np.nan,
                "bounds_f": bounds_f,
                "bounds_i0": lower,
                "bounds_i1": upper,
                "bounds_decision": bounds,
                "t_rho": t_rho,
                "t_bound": t_bound,
                "breusch_godfrey_p_lag1": breusch_godfrey_lag1,
                "breusch_godfrey_p_lag7": breusch_godfrey_lag7,
                "cusum_ratio": stability["cusum_ratio"],
                "cusum_inside": stability["cusum_inside"],
                "cusumsq_ratio": stability["cusumsq_ratio"],
                "cusumsq_inside": stability["cusumsq_inside"],
                "cusum_p": cusum_p,
            }
        )
    return pd.DataFrame(rows)


def _theta(rho, pi, cov, rho_name, pi_name):
    if abs(rho) < 1e-8:
        return np.nan, np.nan
    theta = -pi / rho
    g_rho = pi / rho**2
    g_pi = -1.0 / rho
    variance = (
        g_rho**2 * float(cov.loc[rho_name, rho_name])
        + g_pi**2 * float(cov.loc[pi_name, pi_name])
        + 2 * g_rho * g_pi * float(cov.loc[rho_name, pi_name])
    )
    if variance < 0:
        variance = np.nan
    standard_error = float(np.sqrt(variance)) if np.isfinite(variance) else np.nan
    return float(theta), standard_error


def _ecm_frame(frame, outcome, regressors, exog_cols, p, q):
    y = frame[outcome].astype(float)
    data = pd.DataFrame({"dy": y.diff(), "y_l1": y.shift(1)})
    for i in range(1, int(p)):
        data[f"dy_l{i}"] = data["dy"].shift(i)
    for name in regressors:
        x = frame[name].astype(float)
        data[f"{name}_l1"] = x.shift(1)
        dx = x.diff()
        for j in range(0, int(q)):
            data[f"d_{name}_l{j}"] = dx.shift(j)
    controls = control_matrix(frame, exog_cols).reset_index(drop=True)
    return pd.concat([data.reset_index(drop=True), controls], axis=1).dropna().reset_index(drop=True)


def _recursive_stability(y, design):
    """Brown-Durbin-Evans CUSUM and CUSUMSQ on recursive residuals.

    The CUSUM band is 0.948 (1 + 2 j/n). The CUSUMSQ band is the flat
    half-width 0.114 used in the manuscript. A ratio below 1 means the
    path stays inside that 5% band.
    """
    empty = {"cusum_ratio": np.nan, "cusum_inside": False, "cusumsq_ratio": np.nan, "cusumsq_inside": False}
    residuals = _recursive_residuals(y, design)
    weights = residuals[np.isfinite(residuals)]
    n = len(weights)
    if n < 20 or not np.any(weights):
        return empty
    sigma = np.sqrt(np.sum(weights**2) / n)
    if sigma < 1e-12 or not np.isfinite(sigma):
        return empty
    index = np.arange(1, n + 1, dtype=float)
    cusum = np.cumsum(weights) / (sigma * np.sqrt(n))
    band = 0.948 * (1.0 + 2.0 * (index / n))
    cusum_ratio = float(np.max(np.abs(cusum) / band))
    squared = np.cumsum(weights**2) / np.sum(weights**2)
    halfwidth = 0.114
    cusumsq_ratio = float(np.max(np.abs(squared - index / n)) / halfwidth)
    return {
        "cusum_ratio": cusum_ratio,
        "cusum_inside": bool(cusum_ratio <= 1),
        "cusumsq_ratio": cusumsq_ratio,
        "cusumsq_inside": bool(cusumsq_ratio <= 1),
    }


def _recursive_residuals(y, design):
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    n, k = design.shape
    residuals = np.full(n, np.nan)
    if n <= k:
        return residuals
    cross = design[:k].T @ design[:k]
    moment = design[:k].T @ y[:k]
    ridge = 1e-8 * np.eye(k)
    try:
        beta = np.linalg.solve(cross + ridge, moment)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(cross) @ moment
    for t in range(k, n):
        row = design[t]
        fitted = float(row @ beta)
        try:
            leverage = float(row @ np.linalg.solve(cross + ridge, row))
        except np.linalg.LinAlgError:
            leverage = float(row @ (np.linalg.pinv(cross) @ row))
        scale = max(1.0 + leverage, 1e-12)
        residuals[t] = (y[t] - fitted) / np.sqrt(scale)
        cross = cross + np.outer(row, row)
        moment = moment + row * y[t]
        try:
            beta = np.linalg.solve(cross + ridge, moment)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(cross) @ moment
    return residuals

def error_correction_frame(frame, outcome, regressors, exog_cols, p, q):
    """Conditional error-correction regressors.

    The long-run multiplier uses y_l1 and each regressor's own lag.
    theta = -pi / rho.
    """
    return _ecm_frame(frame, outcome, regressors, exog_cols, p, q)
