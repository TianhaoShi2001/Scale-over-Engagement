"""Misclassification sensitivity for an unlabeled pool that may contain external AIGC (SI-5).

Platform-labeled AIGC is identified from official-tool metadata. The contamination
rate is φ = P(true AIGC | unlabeled). The published calibration is

    φ = (f - FPR) / (TPR - FPR),

with f = 0.242, TPR = 0.693, FPR = 0, and the upper endpoint φ_max = 0.474.
This module does not rerun the detector audit. It applies that scenario to synthetic means.

If a share φ of the observed unlabeled pool has the platform-labeled mean, the implied
mean of the uncontaminated part is

    (observed pool mean - φ * platform-labeled mean) / (1 - φ).

For consumption and exposure comparisons that part is pure human-generated content.
For creator comparisons it is the pure non-adopter benchmark. The break-even hidden
mean is the external-AIGC mean that would make this contrast zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.columns import COMPARISON, FOCAL


def contamination_rate(flag_rate, tpr, fpr):
    denom = tpr - fpr
    if denom <= 0:
        return np.nan
    return float((flag_rate - fpr) / denom)


def mixture_table(pairs, outcomes, phi_grid):
    rows = []
    for outcome in outcomes:
        focal = float(pairs[f"{outcome}{FOCAL}"].mean())
        pool = float(pairs[f"{outcome}{COMPARISON}"].mean())
        for phi in phi_grid:
            pure = _pure_mean(pool, focal, phi)
            hidden = _breakeven_hidden(pool, focal, phi)
            rows.append(
                {
                    "outcome": outcome,
                    "phi": float(phi),
                    "mean_focal": focal,
                    "mean_observed_pool": pool,
                    "observed_contrast": focal - pool,
                    "implied_pure_baseline": pure,
                    "contrast_vs_pure": focal - pure,
                    "breakeven_hidden_mean": hidden,
                }
            )
    return pd.DataFrame(rows)


def _pure_mean(pool, focal, phi):
    if phi >= 1:
        return np.nan
    return (pool - phi * focal) / (1 - phi)


def _breakeven_hidden(pool, focal, phi):
    if phi <= 0 or phi >= 1:
        return np.nan
    return (pool - focal * (1 - phi)) / phi
