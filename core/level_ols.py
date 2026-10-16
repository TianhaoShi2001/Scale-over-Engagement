"""Level OLS for the stationary relative series (Fig. 2(c)).

    R_t = α + β_S S_t + β_E E_t + λ D_t + ε_t

S_t = s_t^a / s_t^u is relative platform-labeled AIGC supply.
E_t is relative consumer engagement, the ratio of mean valid-view rates.
R_t is relative median exposure, the ratio of 31-day cumulative median
exposures (platform-labeled AIGC over unlabeled). D_t is the same control
vector as in the Granger equation. Standard errors are Newey-West with a
7-day bandwidth in the manuscript.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan

from core.exog import control_matrix


def level_ols(frame, outcome, regressors, exog_cols, hac_lags):
    y = frame[outcome].astype(float)
    x_reg = frame[regressors].astype(float)
    controls = control_matrix(frame, exog_cols)
    design = pd.concat([x_reg.reset_index(drop=True), controls.reset_index(drop=True)], axis=1)
    y = y.reset_index(drop=True)
    fit = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    classical = sm.OLS(y, design).fit()
    serial_lag1 = acorr_breusch_godfrey(classical, nlags=1)
    serial_lag7 = acorr_breusch_godfrey(classical, nlags=min(7, len(y) // 5))
    heteroskedasticity = het_breuschpagan(classical.resid, classical.model.exog)
    rows = []
    for name in regressors:
        rows.append(
            {
                "term": name,
                "estimate": float(fit.params[name]),
                "standard_error": float(fit.bse[name]),
                "p_value": float(fit.pvalues[name]),
                "ci95_low": float(fit.conf_int().loc[name, 0]),
                "ci95_high": float(fit.conf_int().loc[name, 1]),
                "breusch_godfrey_p_lag1": float(serial_lag1[1]),
                "breusch_godfrey_p_lag7": float(serial_lag7[1]),
                "breusch_pagan_p": float(heteroskedasticity[1]),
            }
        )
    return pd.DataFrame(rows)


def partial_residuals(frame, outcome, regressors, exog_cols):
    """Residualize R and each regressor on the other regressors plus D_t.

    Fig. 2(c) plots these pairs. The slope drawn on top is the Newey-West
    coefficient from `level_ols`, not a second fit through the scatter.
    """
    controls = control_matrix(frame, exog_cols).reset_index(drop=True)
    outcome_values = frame[outcome].reset_index(drop=True).astype(float)
    rows = []
    for name in regressors:
        others = [column for column in regressors if column != name]
        design = pd.concat(
            [frame[others].reset_index(drop=True).astype(float), controls],
            axis=1,
        )
        residual_outcome = sm.OLS(outcome_values, design).fit().resid
        residual_regressor = sm.OLS(frame[name].reset_index(drop=True).astype(float), design).fit().resid
        for x_value, y_value in zip(residual_regressor, residual_outcome):
            rows.append({"regressor": name, "residual_regressor": float(x_value), "residual_outcome": float(y_value)})
    return pd.DataFrame(rows)
