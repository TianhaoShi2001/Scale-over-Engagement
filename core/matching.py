"""Exact blocking plus nearest-neighbor matching with replacement.

Methods, Behavioral Analyses: categorical covariates are matched exactly;
continuous covariates are matched by Euclidean nearest neighbor after
z-score standardization, with replacement. A caliper is a hard gate outside
the distance. The consumption-side analysis sets that gate at a 3 s
absolute difference in video duration. The focal arm is AIGC-adopting creators,
platform-labeled AIGC, or population-feedback distribution. Pair rows keep `focal_row` and
`comparison_row` so the Abadie--Imbens variance can count how often each
comparison unit is used. That variance is computed in `core/imbens.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.columns import COMPARISON, FOCAL


def match_pairs(focal, comparison, exact_cols, numeric_cols, outcomes, caliper=None):
    """One row per matched focal unit.

    Copied columns are suffixed `_focal` and `_comparison`. A focal unit
    with no finite neighbor inside its exact cell and caliper is dropped.
    """
    pairs, _cells = match_process(focal, comparison, exact_cols, numeric_cols, outcomes, caliper)
    return pairs


def covariate_scale(focal, comparison, columns):
    """Pooled mean and standard deviation used to z-score the continuous covariates."""
    columns = list(columns)
    if not columns:
        return pd.DataFrame(columns=["covariate", "pooled_mean", "pooled_sd"])
    pooled = np.vstack([focal[columns].to_numpy(float), comparison[columns].to_numpy(float)])
    scale = pooled.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return pd.DataFrame({"covariate": columns, "pooled_mean": pooled.mean(axis=0), "pooled_sd": scale})


def exact_cell_counts(focal, comparison, exact_cols):
    """One row per exact cell that contains a focal unit, before the distance is computed."""
    exact_cols = list(exact_cols)
    focal = focal.reset_index(drop=True)
    comparison = comparison.reset_index(drop=True)
    focal_groups = _row_groups(focal, exact_cols)
    comparison_groups = _row_groups(comparison, exact_cols)
    rows = []
    for cell_key, focal_index in focal_groups.items():
        comparison_index = comparison_groups.get(cell_key)
        row = _named_cell(cell_key, exact_cols)
        row["n_focal"] = int(len(focal_index))
        row["n_comparison"] = 0 if comparison_index is None else int(len(comparison_index))
        rows.append(row)
    return pd.DataFrame(rows)


def match_process(focal, comparison, exact_cols, numeric_cols, outcomes, caliper=None):
    """Exact cells, then nearest neighbor. Returns the pairs and a cell ledger.

    The ledger is the match itself: for each exact cell, how many focal units
    went in, how many comparison units were available, how many received a
    finite neighbor, and how many were dropped.
    """
    caliper = caliper or {}
    exact_cols = list(exact_cols)
    numeric_cols = list(numeric_cols)
    focal = focal.reset_index(drop=True)
    comparison = comparison.reset_index(drop=True)
    focal_z, comparison_z = _standardized(focal, comparison, numeric_cols)
    focal_groups = _row_groups(focal, exact_cols)
    comparison_groups = _row_groups(comparison, exact_cols)
    copied_columns = exact_cols + numeric_cols + list(outcomes)
    matched_blocks = []
    ledger = []

    for cell_key, focal_index in focal_groups.items():
        comparison_index = comparison_groups.get(cell_key)
        focal_index = np.asarray(focal_index, dtype=int)
        n_comparison = 0 if comparison_index is None else len(comparison_index)
        n_matched = 0
        median_distance = np.nan
        if n_comparison:
            comparison_index = np.asarray(comparison_index, dtype=int)
            distances = _distances(
                focal,
                comparison,
                focal_z,
                comparison_z,
                focal_index,
                comparison_index,
                numeric_cols,
                caliper,
            )
            neighbor = distances.argmin(axis=1)
            nearest = distances[np.arange(len(focal_index)), neighbor]
            matched = np.isfinite(nearest)
            n_matched = int(matched.sum())
            if n_matched:
                median_distance = float(np.median(nearest[matched]))
                focal_rows = focal_index[matched]
                comparison_rows = comparison_index[neighbor[matched]]
                matched_blocks.append(
                    _pair_block(
                        focal.iloc[focal_rows],
                        comparison.iloc[comparison_rows],
                        copied_columns,
                        nearest[matched],
                        focal_rows,
                        comparison_rows,
                    )
                )
        row = _named_cell(cell_key, exact_cols)
        row.update(
            {
                "n_focal": int(len(focal_index)),
                "n_comparison": int(n_comparison),
                "n_matched": n_matched,
                "n_dropped": int(len(focal_index) - n_matched),
                "median_distance": median_distance,
            }
        )
        ledger.append(row)

    cells = pd.DataFrame(ledger)
    if not matched_blocks:
        empty_columns = ["distance", "focal_row", "comparison_row"]
        empty_columns += [f"{name}{FOCAL}" for name in copied_columns]
        empty_columns += [f"{name}{COMPARISON}" for name in copied_columns]
        return pd.DataFrame(columns=empty_columns), cells
    return pd.concat(matched_blocks, ignore_index=True), cells


def _named_cell(cell_key, exact_cols):
    if not exact_cols:
        return {}
    return {column: cell_key[i] for i, column in enumerate(exact_cols)}


def _distances(focal, comparison, focal_z, comparison_z, focal_index, comparison_index, numeric_cols, caliper):
    if numeric_cols:
        distances = np.linalg.norm(
            focal_z[focal_index, None, :] - comparison_z[comparison_index][None, :, :],
            axis=2,
        )
    else:
        distances = np.zeros((len(focal_index), len(comparison_index)))
    for column, limit in caliper.items():
        gap = np.abs(
            focal.iloc[focal_index][column].to_numpy(float)[:, None]
            - comparison.iloc[comparison_index][column].to_numpy(float)[None, :]
        )
        distances = np.where(gap <= float(limit), distances, np.inf)
    return distances


def _pair_block(focal_rows, comparison_rows, columns, distance, focal_index, comparison_index):
    block = pd.DataFrame(
        {
            "distance": distance,
            "focal_row": np.asarray(focal_index, dtype=int),
            "comparison_row": np.asarray(comparison_index, dtype=int),
        }
    )
    for column in columns:
        block[f"{column}{FOCAL}"] = focal_rows[column].to_numpy()
        block[f"{column}{COMPARISON}"] = comparison_rows[column].to_numpy()
    return block


def _standardized(focal, comparison, columns):
    if not columns:
        return None, None
    scale = covariate_scale(focal, comparison, columns)
    center = scale["pooled_mean"].to_numpy(float)
    spread = scale["pooled_sd"].to_numpy(float)
    focal_z = (focal[list(columns)].to_numpy(float) - center) / spread
    comparison_z = (comparison[list(columns)].to_numpy(float) - center) / spread
    return focal_z, comparison_z


def _row_groups(frame, columns):
    if not columns:
        return {(): np.arange(len(frame))}
    grouped = frame.groupby(list(columns), sort=False).indices
    return {_cell_key(key): index for key, index in grouped.items()}


def _cell_key(key):
    if isinstance(key, tuple):
        return key
    return (key,)
