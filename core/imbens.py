"""Abadie--Imbens variance for the matching estimator of the mean contrast.

One-to-one nearest-neighbor matching with replacement (Abadie and Imbens,
2006). M = 1. For a matched sample of N_1 focal units,

    Var(tau) = (1 / N_1^2) * sum_i w_i * sigma^2(X_i)

A matched focal unit has weight 1. A comparison unit used K times has
weight K^2. Units that are not in the matched sample have weight 0.

sigma^2(X_i) is estimated from up to J nearest neighbors in the same arm
and the same exact-match cell, on the same z-scored covariates as the match:

    sigma^2(X_i) = (J_i / (J_i + 1)) * (Y_i - mean of those neighbors)^2

J is at most 5. The manuscript reports the Hodges-Lehmann median and the
Wilcoxon signed-rank p-value for the paired contrast. Those rank summaries
stay in rank_tests.py. They are not this variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.columns import COMPARISON, FOCAL
from core.matching import _row_groups, _standardized
from core.rank_tests import paired_test


def contrast_table(focal, comparison, pairs, outcomes, exact_cols, numeric_cols, n_neighbors=5):
    """One row per outcome: rank summary, plus the Abadie--Imbens mean contrast."""
    focal = focal.reset_index(drop=True)
    comparison = comparison.reset_index(drop=True)
    focal_z, comparison_z = _standardized(focal, comparison, numeric_cols)
    focal_neighbors = _neighbor_index(focal, focal_z, exact_cols, n_neighbors)
    comparison_neighbors = _neighbor_index(comparison, comparison_z, exact_cols, n_neighbors)
    focal_rows = pairs["focal_row"].to_numpy(int)
    comparison_rows = pairs["comparison_row"].to_numpy(int)
    rows = []
    for name in outcomes:
        rank = paired_test(pairs[f"{name}{FOCAL}"], pairs[f"{name}{COMPARISON}"], name)
        rank["wilcoxon_p"] = rank.pop("p_value")
        variance = _att_inference(
            rank["mean_difference"],
            _sigma2(focal[name].to_numpy(float), focal_neighbors),
            _sigma2(comparison[name].to_numpy(float), comparison_neighbors),
            focal_rows,
            comparison_rows,
        )
        rank.update(variance)
        rows.append(rank)
    return pd.DataFrame(rows)


def _neighbor_index(frame, z, exact_cols, n_neighbors):
    """Row positions of up to n_neighbors same-arm neighbors. -1 if absent."""
    neighbors = np.full((len(frame), int(n_neighbors)), -1, dtype=int)
    if len(frame) == 0:
        return neighbors
    for _, index in _row_groups(frame, exact_cols).items():
        index = np.asarray(index, dtype=int)
        if len(index) < 2:
            continue
        width = min(int(n_neighbors), len(index) - 1)
        if z is None:
            for local, row in enumerate(index):
                others = np.concatenate([index[local + 1 :], index[:local]])[:width]
                neighbors[row, :width] = others
            continue
        block = z[index]
        distance = np.linalg.norm(block[:, None, :] - block[None, :, :], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        np.fill_diagonal(distance, np.inf)
        chosen = np.argpartition(distance, kth=width - 1, axis=1)[:, :width]
        for local, row in enumerate(index):
            local_choice = chosen[local]
            order = np.argsort(distance[local, local_choice])
            neighbors[row, :width] = index[local_choice[order]]
    return neighbors


def _sigma2(outcome, neighbors):
    outcome = np.asarray(outcome, dtype=float)
    valid = neighbors != -1
    neighbor_count = valid.sum(axis=1)
    safe = np.where(valid, neighbors, 0)
    neighbor_outcome = np.where(valid, outcome[safe], 0.0)
    sigma2 = np.full(len(outcome), np.nan)
    has_neighbor = neighbor_count > 0
    neighbor_mean = neighbor_outcome[has_neighbor].sum(axis=1) / neighbor_count[has_neighbor]
    sigma2[has_neighbor] = (neighbor_count[has_neighbor] / (neighbor_count[has_neighbor] + 1.0)) * (
        outcome[has_neighbor] - neighbor_mean
    ) ** 2
    return sigma2


def _att_inference(mean_difference, focal_sigma2, comparison_sigma2, focal_rows, comparison_rows):
    n_focal = len(focal_rows)
    empty = {
        "variance": np.nan,
        "standard_error": np.nan,
        "z_statistic": np.nan,
        "p_value": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
    }
    if n_focal == 0 or not np.isfinite(mean_difference):
        return empty
    times_used = np.bincount(comparison_rows, minlength=len(comparison_sigma2))
    used = times_used > 0
    comparison_term = np.zeros(len(comparison_sigma2), dtype=float)
    comparison_term[used] = (times_used[used] ** 2) * comparison_sigma2[used]
    variance = (np.nansum(focal_sigma2[focal_rows]) + np.nansum(comparison_term)) / n_focal**2
    if not np.isfinite(variance) or variance < 0:
        return empty
    standard_error = float(np.sqrt(variance))
    if standard_error == 0:
        return empty
    z_statistic = float(mean_difference / standard_error)
    p_value = float(2 * (1 - norm.cdf(abs(z_statistic))))
    return {
        "variance": float(variance),
        "standard_error": standard_error,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "ci95_low": float(mean_difference - 1.96 * standard_error),
        "ci95_high": float(mean_difference + 1.96 * standard_error),
    }
