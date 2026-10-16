"""Exogenous controls shared by the Granger, level-OLS, and ARDL fits.

The manuscript control vector D_t contains weekday, holiday, Local Life
campaign-period, and month indicators, plus the log of Local Life
vertical-wide exposure. In this synthetic sample the holiday and campaign
flags are placeholders with those roles. Categorical columns are expanded
to dummies with a reference level removed.
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL = {"weekday", "month"}


def control_matrix(frame, columns):
    parts = [pd.Series(1.0, index=frame.index, name="const")]
    for column in columns:
        series = frame[column]
        if column in CATEGORICAL or series.dtype == object:
            dummies = pd.get_dummies(series.astype(str), prefix=column, drop_first=True, dtype=float)
            parts.append(dummies)
        else:
            parts.append(series.astype(float).rename(column))
    design = pd.concat(parts, axis=1)
    keep = [column for column in design.columns if column == "const" or design[column].std(ddof=0) > 0]
    return design.loc[:, keep]
