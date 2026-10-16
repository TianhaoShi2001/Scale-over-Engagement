"""Run the manuscript pipeline on the synthetic tables.

Each public function is one analysis step. `run_all` writes the synthetic
tables, the result tables, and a timestamped log. The figure notebooks in
`visualizer/` read those files and draw Figure 1 and Figure 2.
"""

from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.ardl import ardl_equation
from core.balance import balance_before_after, propensity_overlap
from core.cinelli import robustness_table
from core.granger import granger_test
from core.imbens import contrast_table
from core.integration import integration_table
from core.make_synthetic import generate
from core.matching import match_pairs
from core.level_ols import level_ols, partial_residuals
from core.misclassification import contamination_rate, mixture_table
from core.columns import (
    COMPARISON,
    EXPOSURE_SHARES,
    FIRST_DAY_MEDIAN_EXPOSURE,
    FOCAL,
    LABELED_SUPPLY,
    LOG_FULL_VIEW_RATE,
    LOG_LABELED_SUPPLY,
    LOG_UNLABELED_SUPPLY,
    LOG_VALID_VIEW_RATE,
    UNLABELED_SUPPLY,
)


def run_all(root=None, rebuild=True):
    root = Path(root) if root is not None else find_root()
    cfg = load_config(root)
    frames = ensure_synthetic(root, cfg, rebuild=rebuild)
    log = []
    log.append(f"root {root}")
    log.append(f"time {datetime.now().isoformat(timespec='seconds')}")
    log.append(f"python {platform.python_version()}")
    log.append(f"seed {cfg['seed']}")
    log.append(
        "Fig. 2(d) is the long-run elasticity of ln(e_t) with respect to ln(s_t^a). "
        "Fig. 2(e)(f) compare population-feedback and individual-feedback distribution."
    )

    summaries = []
    for name in ("creator", "consumer", "video", "algorithm"):
        out = run_matched_task(root, cfg, frames, name)
        summaries.append(out["summary"])
        log.append(
            f"match {name}: pairs {out['summary']['n_pairs']} "
            f"of {out['summary']['n_focal']} focal units"
        )
    log.append(
        "matching SE: Abadie-Imbens, "
        f"within-arm neighbors {int(cfg.get('imbens_neighbors', 5))}"
    )
    pd.DataFrame(summaries).to_csv(_results(root) / "match_summary.csv", index=False)

    granger = run_granger(root, cfg, frames["daily_panel"])
    log.append(
        f"granger lag {cfg['granger_lag']}: F {granger['f_statistic'].iloc[0]:.3f} "
        f"bootstrap p {granger['bootstrap_p'].iloc[0]:.3f}"
    )
    tiers = run_exposure_tiers(root, cfg, frames["daily_panel"])
    engage = run_engagement(root, cfg, frames["daily_panel"])
    ols = run_level_ols(root, cfg, frames["daily_panel"])
    log.append(
        "level OLS "
        + ", ".join(f"{r.term} {r.estimate:.3f}" for r in ols.itertuples())
    )
    log.append(
        "engagement theta "
        + ", ".join(
            f"{r.outcome} {r.theta:.3f}"
            for r in engage[engage.regressor == LOG_LABELED_SUPPLY].itertuples()
        )
    )
    orders = run_integration(root, cfg, frames["daily_panel"])
    log.append("integration " + ", ".join(f"{r.series}:{r.order}" for r in orders.itertuples()))

    cinelli = run_cinelli(root, cfg, frames)
    misclass = run_misclassification(root, cfg, frames)
    caliper = run_duration_caliper(root, cfg, frames)
    lag_grid = run_granger_sensitivity(root, cfg, frames["daily_panel"])
    ardl_grid = run_ardl_sensitivity(root, cfg, frames["daily_panel"])
    log.append(f"cinelli rows {len(cinelli)}")
    log.append(f"misclassification rows {len(misclass)}")
    log.append(f"caliper rows {len(caliper)}")
    log.append(f"granger sensitivity rows {len(lag_grid)}")
    log.append(f"ardl sensitivity rows {len(ardl_grid)}")
    point_phi = contamination_rate(cfg["phi_flag_rate"], cfg["phi_tpr"], cfg["phi_fpr"])
    log.append(f"phi point estimate {point_phi:.3f}; configured upper scenario {cfg['phi_max']}")

    soei = np.log(frames["daily_panel"]["rel_supply"] / frames["daily_panel"]["rel_engagement"])
    pd.DataFrame(
        [{"n_days": len(soei), "share_soei_positive": float((soei > 0).mean()), "median_soei": float(soei.median())}]
    ).to_csv(_results(root) / "soei_summary.csv", index=False)
    log.append("tables are in data/results. visualizer/ draws Figure 1 and Figure 2 from those files.")
    _write_log(root, log)
    print("\n".join(log))
    return log


def run_creator():
    return _notebook_task("creator")


def run_consumer():
    return _notebook_task("consumer")


def run_video():
    return _notebook_task("video")


def run_algorithm():
    return _notebook_task("algorithm")


def run_granger_notebook():
    root, cfg, frames = _prepare()
    return run_granger(root, cfg, frames["daily_panel"])


def run_tiers_notebook():
    root, cfg, frames = _prepare()
    return run_exposure_tiers(root, cfg, frames["daily_panel"])


def run_ols_notebook():
    root, cfg, frames = _prepare()
    return run_level_ols(root, cfg, frames["daily_panel"])


def run_engagement_notebook():
    root, cfg, frames = _prepare()
    return run_engagement(root, cfg, frames["daily_panel"])


def run_integration_notebook():
    root, cfg, frames = _prepare()
    return run_integration(root, cfg, frames["daily_panel"])


def run_cinelli_notebook():
    root, cfg, frames = _prepare()
    return run_cinelli(root, cfg, frames)


def run_misclassification_notebook():
    root, cfg, frames = _prepare()
    return run_misclassification(root, cfg, frames)


def run_caliper_notebook():
    root, cfg, frames = _prepare()
    return run_duration_caliper(root, cfg, frames)


def run_granger_sensitivity_notebook():
    root, cfg, frames = _prepare()
    return run_granger_sensitivity(root, cfg, frames["daily_panel"])


def run_ardl_sensitivity_notebook():
    root, cfg, frames = _prepare()
    return run_ardl_sensitivity(root, cfg, frames["daily_panel"])


def find_root():
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "run_pipeline.py").exists() and (cand / "core" / "analysis.yaml").exists():
            return cand
    raise FileNotFoundError("Could not find run_pipeline.py. Start the notebook from the repository.")


def load_config(root):
    with open(Path(root) / "core" / "analysis.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_synthetic(root, cfg, rebuild=False):
    folder = Path(root) / "data" / "synthetic"
    marker = folder / "daily_panel.csv"
    if marker.exists() and not rebuild:
        names = ["creators", "consumers", "videos", "algorithm", "daily_panel"]
        return {name: pd.read_csv(folder / f"{name}.csv") for name in names}
    frames = generate(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    return frames


def run_matched_task(root, cfg, frames, name):
    spec = cfg["tasks"][name]
    pool = frames[_pool_name(name)]
    focal = pool[pool["treated"] == 1].copy()
    comparison = pool[pool["treated"] == 0].copy()
    pairs = match_pairs(
        focal,
        comparison,
        exact_cols=spec["exact"],
        numeric_cols=spec["numeric"],
        outcomes=spec["outcomes"],
        caliper=spec.get("caliper") or {},
    )
    _assert_protocol(pairs, spec)
    effects = contrast_table(
        focal,
        comparison,
        pairs,
        spec["outcomes"],
        spec["exact"],
        spec["numeric"],
        n_neighbors=int(cfg.get("imbens_neighbors", 5)),
    )
    covariates = list(spec["exact"]) + list(spec["numeric"])
    balance = balance_before_after(focal, comparison, pairs, covariates)
    after_focal = pd.DataFrame({column: pairs[f"{column}{FOCAL}"] for column in spec["numeric"]})
    after_comparison = pd.DataFrame({column: pairs[f"{column}{COMPARISON}"] for column in spec["numeric"]})
    summary = {
        "task": name,
        "figure": spec["figure"],
        "n_focal": int(len(focal)),
        "n_pairs": int(len(pairs)),
        "overlap_before": propensity_overlap(focal, comparison, spec["numeric"]),
        "overlap_after": propensity_overlap(after_focal, after_comparison, spec["numeric"]),
    }
    folder = _results(root)
    pairs.to_csv(folder / f"{name}_pairs.csv", index=False)
    effects.to_csv(folder / f"{name}_effects.csv", index=False)
    balance.to_csv(folder / f"{name}_balance.csv", index=False)
    return {"pairs": pairs, "effects": effects, "balance": balance, "summary": summary}


def run_granger(root, cfg, daily, lag=None, n_boot=None, outcome=None, focal=None, control=None):
    table = granger_test(
        daily,
        outcome or FIRST_DAY_MEDIAN_EXPOSURE,
        focal or LABELED_SUPPLY,
        control or UNLABELED_SUPPLY,
        cfg["time_exog"],
        lag if lag is not None else cfg["granger_lag"],
        cfg["hac_lags"],
        n_boot if n_boot is not None else cfg["bootstrap_replications"],
        cfg["block_length"],
        int(cfg["seed"]),
    )
    if outcome is None:
        table.to_csv(_results(root) / "granger_baseline.csv", index=False)
    return table


def run_exposure_tiers(root, cfg, daily):
    parts = [
        ardl_equation(
            daily,
            outcome,
            [LABELED_SUPPLY, UNLABELED_SUPPLY],
            cfg["time_exog"],
            cfg["ardl_p"],
            cfg["ardl_q"],
            cfg["hac_lags"],
        )
        for outcome in EXPOSURE_SHARES
    ]
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(_results(root) / "ardl_tiers.csv", index=False)
    return table


def run_engagement(root, cfg, daily):
    """Fig. 2(d): long-run elasticity of ln(e_t) with respect to ln(s_t^a)."""
    parts = [
        ardl_equation(
            daily,
            outcome,
            [LOG_LABELED_SUPPLY, LOG_UNLABELED_SUPPLY],
            cfg["time_exog"],
            cfg["ardl_p"],
            cfg["ardl_q"],
            cfg["hac_lags"],
        )
        for outcome in (LOG_FULL_VIEW_RATE, LOG_VALID_VIEW_RATE)
    ]
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(_results(root) / "ardl_engagement.csv", index=False)
    return table


def run_level_ols(root, cfg, daily):
    table = level_ols(
        daily,
        "rel_exposure",
        ["rel_supply", "rel_engagement"],
        cfg["time_exog"],
        cfg["hac_lags"],
    )
    table.to_csv(_results(root) / "level_ols.csv", index=False)
    partial_residuals(
        daily,
        "rel_exposure",
        ["rel_supply", "rel_engagement"],
        cfg["time_exog"],
    ).to_csv(_results(root) / "level_ols_partial.csv", index=False)
    return table


def run_integration(root, cfg, daily):
    table = integration_table(daily, cfg["integration_series"])
    table.to_csv(_results(root) / "integration_order.csv", index=False)
    return table


def run_cinelli(root, cfg, frames):
    parts = []
    for name in ("creator", "consumer", "video", "algorithm"):
        spec = cfg["tasks"][name]
        pairs = pd.read_csv(_results(root) / f"{name}_pairs.csv")
        table = robustness_table(pairs, spec["outcomes"], spec["numeric"])
        table.insert(0, "task", name)
        parts.append(table)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(_results(root) / "cinelli.csv", index=False)
    return out


def run_misclassification(root, cfg, frames):
    parts = []
    for name in ("creator", "consumer", "video"):
        spec = cfg["tasks"][name]
        pairs = pd.read_csv(_results(root) / f"{name}_pairs.csv")
        table = mixture_table(pairs, spec["outcomes"], cfg["phi_grid"])
        table.insert(0, "task", name)
        parts.append(table)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(_results(root) / "misclassification.csv", index=False)
    return out


def run_duration_caliper(root, cfg, frames):
    spec = cfg["tasks"]["consumer"]
    pool = frames["consumers"]
    focal = pool[pool["treated"] == 1]
    comparison = pool[pool["treated"] == 0]
    rows = []
    for width in cfg["duration_calipers"]:
        caliper = {"duration_seconds": float(width)}
        pairs = match_pairs(
            focal,
            comparison,
            spec["exact"],
            spec["numeric"],
            ["valid_view_rate"],
            caliper,
        )
        effect = contrast_table(
            focal,
            comparison,
            pairs,
            ["valid_view_rate"],
            spec["exact"],
            spec["numeric"],
            n_neighbors=int(cfg.get("imbens_neighbors", 5)),
        ).iloc[0].to_dict()
        effect["caliper_seconds"] = float(width)
        rows.append(effect)
    table = pd.DataFrame(rows)
    table.to_csv(_results(root) / "duration_caliper.csv", index=False)
    return table


def run_granger_sensitivity(root, cfg, daily):
    rows = []
    specs = [
        ("labeled_supply", FIRST_DAY_MEDIAN_EXPOSURE, LABELED_SUPPLY, UNLABELED_SUPPLY),
        ("reverse_exposure", LABELED_SUPPLY, FIRST_DAY_MEDIAN_EXPOSURE, UNLABELED_SUPPLY),
        ("unlabeled_supply", FIRST_DAY_MEDIAN_EXPOSURE, UNLABELED_SUPPLY, LABELED_SUPPLY),
    ]
    for lag in cfg["granger_lag_grid"]:
        for label, outcome, focal, control in specs:
            if label != "labeled_supply" and lag != cfg["granger_lag"]:
                continue
            table = granger_test(
                daily,
                outcome,
                focal,
                control,
                cfg["time_exog"],
                lag,
                cfg["hac_lags"],
                cfg["bootstrap_replications_grid"],
                cfg["block_length"],
                int(cfg["seed"]) + int(lag),
            )
            rows.append(
                {
                    "spec": label,
                    "lag": lag,
                    "f_statistic": float(table["f_statistic"].iloc[0]),
                    "bootstrap_p": float(table["bootstrap_p"].iloc[0]),
                    "mean_focal_coef": float(table["estimate"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(_results(root) / "granger_sensitivity.csv", index=False)
    return out


def run_ardl_sensitivity(root, cfg, daily):
    rows = []
    equations = [
        (LOG_FULL_VIEW_RATE, [LOG_LABELED_SUPPLY, LOG_UNLABELED_SUPPLY]),
        (LOG_VALID_VIEW_RATE, [LOG_LABELED_SUPPLY, LOG_UNLABELED_SUPPLY]),
        (EXPOSURE_SHARES[0], [LABELED_SUPPLY, UNLABELED_SUPPLY]),
    ]
    for p in cfg["ardl_lag_grid"]:
        for q in cfg["ardl_lag_grid"]:
            for outcome, regressors in equations:
                try:
                    table = ardl_equation(
                        daily, outcome, regressors, cfg["time_exog"], p, q, cfg["hac_lags"]
                    )
                except Exception as exc:
                    rows.append({"outcome": outcome, "p": p, "q": q, "error": str(exc)})
                    continue
                focal = regressors[0]
                hit = table[table["regressor"] == focal].iloc[0]
                rows.append(
                    {
                        "outcome": outcome,
                        "regressor": focal,
                        "p": p,
                        "q": q,
                        "theta": hit["theta"],
                        "rho": hit["rho"],
                        "t_rho": hit["t_rho"],
                        "t_bound": hit["t_bound"],
                        "bounds_f": hit["bounds_f"],
                        "bounds_decision": hit["bounds_decision"],
                        "breusch_godfrey_p_lag1": hit["breusch_godfrey_p_lag1"],
                        "breusch_godfrey_p_lag7": hit["breusch_godfrey_p_lag7"],
                        "cusum_ratio": hit["cusum_ratio"],
                        "cusumsq_ratio": hit["cusumsq_ratio"],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(_results(root) / "ardl_lag_sensitivity.csv", index=False)
    return out


def _notebook_task(name):
    root, cfg, frames = _prepare()
    return run_matched_task(root, cfg, frames, name)


def _prepare():
    root = find_root()
    cfg = load_config(root)
    frames = ensure_synthetic(root, cfg, rebuild=False)
    return root, cfg, frames


def _pool_name(task):
    return {"creator": "creators", "consumer": "consumers", "video": "videos", "algorithm": "algorithm"}[task]


def _assert_protocol(pairs, spec):
    if pairs.empty:
        raise RuntimeError(f"No pairs for {spec['figure']}. Exact cells or the caliper removed every unit.")
    for column in spec["exact"]:
        same = pairs[f"{column}{FOCAL}"].astype(str).to_numpy() == pairs[f"{column}{COMPARISON}"].astype(str).to_numpy()
        if not same.all():
            raise RuntimeError(f"Exact match failed on {column}.")
    for column, limit in (spec.get("caliper") or {}).items():
        gap = (pairs[f"{column}{FOCAL}"] - pairs[f"{column}{COMPARISON}"]).abs()
        if (gap > float(limit) + 1e-6).any():
            raise RuntimeError(f"Caliper failed on {column}.")


def _results(root):
    folder = Path(root) / "data" / "results"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_log(root, lines):
    folder = Path(root) / "data" / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"run_{stamp}.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
