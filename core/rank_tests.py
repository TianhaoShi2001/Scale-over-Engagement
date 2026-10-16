"""Paired Wilcoxon test and Hodges-Lehmann median difference.

The manuscript reports the Hodges-Lehmann (HL) median of the paired
differences, with a Wilcoxon signed-rank p-value. That p-value is a rank
summary. It is not the variance of the matching estimator. The mean
contrast uses the Abadie--Imbens variance in core/imbens.py.
The difference is the focal arm minus the comparison arm: AIGC-adopting
minus non-adopting, platform-labeled AIGC minus unlabeled content, or
population-feedback minus individual-feedback.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.columns import COMPARISON, FOCAL


def hodges_lehmann(differences):
    """Median of the Walsh averages (d_i + d_j) / 2 for i <= j."""
    ordered = np.sort(np.asarray(differences, dtype=float))
    ordered = ordered[np.isfinite(ordered)]
    n_pairs = len(ordered)
    if n_pairs == 0:
        return np.nan
    left, right = np.triu_indices(n_pairs)
    return float(np.median((ordered[left] + ordered[right]) / 2.0))


def paired_test(focal_values, comparison_values, name):
    difference = np.asarray(focal_values, dtype=float) - np.asarray(comparison_values, dtype=float)
    try:
        statistic, p_value = stats.wilcoxon(
            difference, zero_method="wilcox", alternative="two-sided", method="auto"
        )
        statistic, p_value = float(statistic), float(p_value)
    except ValueError:
        statistic, p_value = np.nan, np.nan
    return {
        "outcome": name,
        "n_pairs": int(np.isfinite(difference).sum()),
        "hl_focal_minus_comparison": hodges_lehmann(difference),
        "mean_difference": float(np.nanmean(difference)),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def paired_tests(pairs, outcomes):
    return [
        paired_test(pairs[f"{name}{FOCAL}"], pairs[f"{name}{COMPARISON}"], name)
        for name in outcomes
    ]
