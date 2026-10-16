"""Synthetic tables for the pipeline.

Column roles follow data/schema.yaml. These tables are synthetic.
They are not the platform logs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.columns import (
    EXPOSURE_SHARES,
    FIRST_DAY_MEDIAN_EXPOSURE,
    LABELED_SUPPLY,
    LOG_FULL_VIEW_RATE,
    LOG_LABELED_SUPPLY,
    LOG_UNLABELED_SUPPLY,
    LOG_VALID_VIEW_RATE,
    UNLABELED_SUPPLY,
)


def generate(cfg):
    rng = np.random.default_rng(int(cfg["seed"]))
    sizes = cfg["sizes"]
    pools = {
        "creators": _creators(sizes["creator"], rng),
        "consumers": _consumers(sizes["consumer"], rng),
        "videos": _videos(sizes["video"], rng),
        "algorithm": _algorithm(sizes["algorithm"], rng),
        "daily_panel": _daily(int(cfg["n_days"]), rng),
    }
    return pools


def _creators(n_arm, rng):
    arms = [_creator_arm(n_arm, rng, treated=1), _creator_arm(n_arm, rng, treated=0)]
    return pd.concat(arms, ignore_index=True)


def _creator_arm(n, rng, treated):
    frame = pd.DataFrame(
        {
            "treated": treated,
            "category": rng.choice(["food", "beauty", "travel", "leisure"], n),
            "city": rng.choice(["north", "east", "south"], n),
            "brand_tier": rng.choice(["tier_1", "tier_2", "tier_3"], n),
            "uploads_14d": rng.lognormal(1.1, 0.35, n),
            "uploads_60d": rng.lognormal(2.0, 0.35, n),
            "plays_14d": rng.lognormal(4.5, 0.45, n),
            "plays_60d": rng.lognormal(5.6, 0.45, n),
            "watch_time_14d": rng.lognormal(5.2, 0.40, n),
            "watch_time_60d": rng.lognormal(6.3, 0.40, n),
            "followers": rng.lognormal(6.5, 0.8, n),
        }
    )
    # Volume shifts with adoption. Aggregate views stay close across arms.
    frame["video_volume"] = (
        6 + 0.35 * frame["uploads_14d"] + (14 if treated else 0) + rng.normal(0, 3.0, n)
    )
    frame["total_valid_views"] = (
        400 + 8 * np.log1p(frame["plays_14d"]) + rng.normal(0, 60, n) + (8 if treated else 0)
    )
    frame["total_full_views"] = (
        70 + 2 * np.log1p(frame["plays_14d"]) + rng.normal(0, 18, n)
    )
    for column in ("video_volume", "total_valid_views", "total_full_views"):
        frame[column] = frame[column].clip(lower=0)
    return frame


def _consumers(n_arm, rng):
    arms = [_consumer_arm(n_arm, rng, treated=1), _consumer_arm(n_arm, rng, treated=0)]
    return pd.concat(arms, ignore_index=True)


def _consumer_arm(n, rng, treated):
    frame = pd.DataFrame(
        {
            "treated": treated,
            "gender": rng.choice(["female", "male"], n),
            "city": rng.choice(["north", "east", "south"], n),
            "category": rng.choice(["food", "beauty", "leisure"], n),
            "time_of_day": rng.choice(["morning", "noon", "evening", "night"], n),
            "upload_week": rng.choice(["week_1", "week_2", "week_3", "week_4"], n),
            "age": rng.normal(32, 8, n).clip(16, 70),
            "duration_seconds": rng.normal(18, 6, n).clip(5, 60),
            "plays_labeled_14d": rng.lognormal(2.2, 0.5, n),
            "plays_labeled_60d": rng.lognormal(3.2, 0.5, n),
            "plays_unlabeled_14d": rng.lognormal(3.4, 0.5, n),
            "plays_unlabeled_60d": rng.lognormal(4.4, 0.5, n),
            "watch_time_labeled_14d": rng.lognormal(3.0, 0.5, n),
            "watch_time_labeled_60d": rng.lognormal(4.0, 0.5, n),
            "watch_time_unlabeled_14d": rng.lognormal(4.0, 0.5, n),
            "watch_time_unlabeled_60d": rng.lognormal(5.0, 0.5, n),
            "creator_uploads_14d": rng.lognormal(1.2, 0.4, n),
            "creator_uploads_60d": rng.lognormal(2.1, 0.4, n),
            "creator_plays_14d": rng.lognormal(4.4, 0.4, n),
            "creator_plays_60d": rng.lognormal(5.5, 0.4, n),
            "creator_watch_time_14d": rng.lognormal(5.0, 0.4, n),
            "creator_watch_time_60d": rng.lognormal(6.1, 0.4, n),
            "creator_followers": rng.lognormal(6.4, 0.7, n),
        }
    )
    # Labeled interactions draw lower engagement. Duration enters that level.
    gap = 0.06 if treated else 0.0
    duration_term = 0.002 * (frame["duration_seconds"] - 18)
    frame["valid_view_rate"] = (0.46 - duration_term - gap + rng.normal(0, 0.04, n)).clip(0.02, 0.95)
    frame["full_view_rate"] = (0.18 - 0.4 * gap + rng.normal(0, 0.03, n)).clip(0.01, 0.9)
    frame["early_retention"] = (0.62 - 0.5 * gap + rng.normal(0, 0.04, n)).clip(0.05, 0.98)
    frame["view_fraction"] = (0.40 - 0.7 * gap + rng.normal(0, 0.05, n)).clip(0.02, 0.99)
    return frame


def _videos(n_arm, rng):
    arms = [_video_arm(n_arm, rng, treated=1), _video_arm(n_arm, rng, treated=0)]
    return pd.concat(arms, ignore_index=True)


def _video_arm(n, rng, treated):
    frame = pd.DataFrame(
        {
            "treated": treated,
            "category": rng.choice(["food", "beauty", "travel", "leisure"], n),
            "city": rng.choice(["north", "east", "south"], n),
            "upload_week": rng.choice(["week_1", "week_2", "week_3", "week_4"], n),
            "uploads_14d": rng.lognormal(1.2, 0.35, n),
            "uploads_60d": rng.lognormal(2.1, 0.35, n),
            "plays_14d": rng.lognormal(4.5, 0.4, n),
            "plays_60d": rng.lognormal(5.6, 0.4, n),
            "watch_time_14d": rng.lognormal(5.1, 0.4, n),
            "watch_time_60d": rng.lognormal(6.2, 0.4, n),
            "followers": rng.lognormal(6.5, 0.7, n),
        }
    )
    exposure_gap = 180 if treated else 0
    frame["exposure_31d"] = (
        520 + 15 * np.log1p(frame["followers"]) - exposure_gap + rng.normal(0, 40, n)
    ).clip(1, None)
    frame["days_to_90"] = (14 - (5 if treated else 0) + rng.normal(0, 1.5, n)).clip(1, 31)
    return frame


def _algorithm(n_arm, rng):
    arms = [
        _algorithm_arm(n_arm, rng, treated=1),
        _algorithm_arm(n_arm, rng, treated=0),
    ]
    return pd.concat(arms, ignore_index=True)


def _algorithm_arm(n, rng, treated):
    frame = pd.DataFrame(
        {
            "treated": treated,
            "city": rng.choice(["north", "east", "south"], n),
            "gender": rng.choice(["female", "male"], n),
            "age": rng.normal(31, 8, n).clip(16, 70),
            "plays_labeled_14d": rng.lognormal(2.3, 0.45, n),
            "plays_labeled_60d": rng.lognormal(3.3, 0.45, n),
            "plays_unlabeled_14d": rng.lognormal(3.5, 0.45, n),
            "plays_unlabeled_60d": rng.lognormal(4.5, 0.45, n),
            "watch_time_labeled_14d": rng.lognormal(3.1, 0.45, n),
            "watch_time_labeled_60d": rng.lognormal(4.1, 0.45, n),
            "watch_time_unlabeled_14d": rng.lognormal(4.1, 0.45, n),
            "watch_time_unlabeled_60d": rng.lognormal(5.1, 0.45, n),
        }
    )
    # Population feedback (treated = 1) allocates a smaller labeled share.
    # The individual-minus-population gap widens slightly across horizons.
    base = 0.24 + rng.normal(0, 0.02, n)
    gaps = {
        "labeled_exposure_share": 0.035,
        "share_week_1": 0.020,
        "share_week_2": 0.025,
        "share_week_4": 0.030,
        "share_week_8": 0.034,
    }
    for name, gap in gaps.items():
        shift = gap if treated else 0.0
        frame[name] = (base - shift + rng.normal(0, 0.01, n)).clip(0.01, 0.8)
    return frame


def _daily(n_days, rng):
    """Daily series with the signs used in Fig. 2(b)-(d).

    s_t^u is a random walk, so the integration screen has an I(1) series
    to classify. S_t, E_t, and R_t are stationary, which is the level-OLS
    case in the manuscript.
    """
    dates = pd.date_range("2024-06-01", periods=n_days, freq="D")
    unlabeled_supply = np.cumsum(rng.normal(0, 4, n_days)) + 900
    unlabeled_supply = np.clip(unlabeled_supply, 200, None)
    labeled_supply = np.empty(n_days)
    labeled_supply[0] = 260
    for day in range(1, n_days):
        labeled_supply[day] = 260 + 0.65 * (labeled_supply[day - 1] - 260) + rng.normal(0, 12)
    labeled_supply = np.clip(labeled_supply, 40, None)

    supply_change = np.diff(labeled_supply, prepend=labeled_supply[0])
    exposure_change = np.zeros(n_days)
    for day in range(2, n_days):
        exposure_change[day] = (
            0.15 * exposure_change[day - 1]
            - 0.09 * supply_change[day - 1]
            - 0.05 * supply_change[day - 2]
            + rng.normal(0, 1.0)
        )
    first_day_median_exposure = np.clip(35 + np.cumsum(exposure_change), 1, None)

    rel_supply = np.empty(n_days)
    rel_engagement = np.empty(n_days)
    rel_supply[0], rel_engagement[0] = 1.35, 0.82
    for day in range(1, n_days):
        rel_supply[day] = 1.35 + 0.55 * (rel_supply[day - 1] - 1.35) + rng.normal(0, 0.04)
        rel_engagement[day] = 0.82 + 0.50 * (rel_engagement[day - 1] - 0.82) + rng.normal(0, 0.02)
    rel_exposure = (
        1.05
        - 0.75 * (rel_supply - 1.35)
        + 1.50 * (rel_engagement - 0.82)
        + rng.normal(0, 0.04, n_days)
    )

    log_labeled_supply = np.log(labeled_supply)
    log_unlabeled_supply = np.log(unlabeled_supply)
    log_full_view = np.empty(n_days)
    log_valid_view = np.empty(n_days)
    log_full_view[0] = np.log(0.12)
    log_valid_view[0] = np.log(0.34)
    for day in range(1, n_days):
        full_view_target = np.log(0.12) + 0.04 * (log_labeled_supply[day] - log_labeled_supply.mean())
        valid_view_target = np.log(0.34) + 0.03 * (log_labeled_supply[day] - log_labeled_supply.mean())
        log_full_view[day] = log_full_view[day - 1] - 0.45 * (log_full_view[day - 1] - full_view_target) + rng.normal(0, 0.008)
        log_valid_view[day] = log_valid_view[day - 1] - 0.45 * (log_valid_view[day - 1] - valid_view_target) + rng.normal(0, 0.008)

    # Softmax shares w_k,t. The ≤10 tier rises with s_t^a; the 101-1,000
    # tier falls. The other two tiers are weakly related.
    centered_supply = (labeled_supply - labeled_supply.mean()) / labeled_supply.std()
    logits = np.column_stack(
        [
            0.2 + 0.55 * centered_supply + rng.normal(0, 0.15, n_days),
            0.6 + 0.05 * centered_supply + rng.normal(0, 0.15, n_days),
            0.4 - 0.45 * centered_supply + rng.normal(0, 0.15, n_days),
            -0.4 - 0.05 * centered_supply + rng.normal(0, 0.15, n_days),
        ]
    )
    logits -= logits.max(axis=1, keepdims=True)
    shares = np.exp(logits)
    shares /= shares.sum(axis=1, keepdims=True)

    vertical_exposure = np.exp(np.log(1e6) + 0.2 * centered_supply + rng.normal(0, 0.05, n_days))
    holiday = ((dates.month == 10) & (dates.day == 1)).astype(int)
    campaign = ((dates >= "2024-11-01") & (dates <= "2024-11-11")).astype(int)

    return pd.DataFrame(
        {
            "date": dates.astype(str),
            LABELED_SUPPLY: labeled_supply,
            UNLABELED_SUPPLY: unlabeled_supply,
            FIRST_DAY_MEDIAN_EXPOSURE: first_day_median_exposure,
            "rel_supply": rel_supply,
            "rel_engagement": rel_engagement,
            "rel_exposure": rel_exposure,
            EXPOSURE_SHARES[0]: shares[:, 0],
            EXPOSURE_SHARES[1]: shares[:, 1],
            EXPOSURE_SHARES[2]: shares[:, 2],
            EXPOSURE_SHARES[3]: shares[:, 3],
            LOG_FULL_VIEW_RATE: log_full_view,
            LOG_VALID_VIEW_RATE: log_valid_view,
            LOG_LABELED_SUPPLY: log_labeled_supply,
            LOG_UNLABELED_SUPPLY: log_unlabeled_supply,
            "log_vertical_exposure": np.log(vertical_exposure),
            "weekday": dates.day_name(),
            "month": dates.month.astype(int),
            "holiday": holiday,
            "campaign": campaign,
        }
    )
