"""Integration-order screen.

Level Augmented Dickey-Fuller and Phillips-Perron tests, both with an
intercept. When those two disagree at 5%, the KPSS test decides.
KPSS has stationarity as its null. I(0) means stationary in levels.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

try:
    from statsmodels.tsa.stattools import phillips_perron
except ImportError:  # pragma: no cover - depends on the local statsmodels build
    phillips_perron = None


def integration_table(frame, columns, alpha=0.05):
    rows = [_classify(frame[column].to_numpy(float), column, alpha) for column in columns]
    return pd.DataFrame(rows)


def _classify(values, name, alpha):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    adf_p = _adf_p(values)
    pp_p = _pp_p(values)
    kpss_p = _kpss_p(values)
    adf_stationary = adf_p < alpha
    if pp_p is None or (pp_p < alpha) == adf_stationary:
        level_stationary = adf_stationary
        rule = "adf" if pp_p is None else "adf_pp_agree"
    else:
        # KPSS null is stationarity, so a large p-value keeps I(0).
        level_stationary = kpss_p >= alpha
        rule = "kpss"
    if level_stationary:
        order = "I(0)"
    else:
        order = "I(1)" if _adf_p(np.diff(values)) < alpha else "above I(1)"
    return {
        "series": name,
        "adf_p": adf_p,
        "pp_p": pp_p if pp_p is not None else np.nan,
        "kpss_p": kpss_p,
        "decision_rule": rule,
        "order": order,
    }


def _adf_p(values):
    if len(values) < 12:
        return np.nan
    return float(adfuller(values, autolag="AIC", regression="c")[1])


def _pp_p(values):
    if len(values) < 12:
        return None
    if phillips_perron is not None:
        try:
            return float(phillips_perron(values)[1])
        except Exception:
            pass
    return _phillips_perron_intercept(values)


def _phillips_perron_intercept(values):
    """Phillips-Perron Z_tau p-value, intercept only, when statsmodels has no PP routine."""
    try:
        from statsmodels.tsa.adfvalues import mackinnonp
    except Exception:
        return None
    series = np.asarray(values, float)
    change = np.diff(series)
    lagged = series[:-1]
    design = np.column_stack([np.ones(len(change)), lagged])
    coefficient, _, _, _ = np.linalg.lstsq(design, change, rcond=None)
    residual = change - design @ coefficient
    n = len(change)
    lags = max(int(np.floor(4 * (n / 100.0) ** 0.25)), 1)
    sigma2 = float(np.dot(residual, residual) / (n - 2))
    gamma0 = float(np.dot(residual, residual) / n)
    long_run = gamma0
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = float(np.dot(residual[lag:], residual[:-lag]) / n)
        long_run += 2 * weight * gamma
    if sigma2 <= 0 or long_run <= 0:
        return None
    se = float(np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1]))
    if se <= 0:
        return None
    t_stat = float(coefficient[1] / se)
    z_tau = np.sqrt(sigma2 / long_run) * t_stat - 0.5 * (long_run - sigma2) * (n * se) / (np.sqrt(sigma2) * np.sqrt(long_run))
    try:
        return float(mackinnonp(z_tau, regression="c", N=1))
    except Exception:
        return None


def _kpss_p(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat_p = kpss(values, regression="c", nlags="auto")
    return float(stat_p[1])
