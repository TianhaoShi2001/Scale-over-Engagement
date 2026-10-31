# Scale over Engagement: pipeline check on synthetic data

Code for the matching, rank, Granger, level-OLS, and ARDL analyses in *Scale over Engagement*. The raw platform logs cannot be shared. This repository ships a synthetic sample with the same column roles, so a reader can run the pipeline. A successful run writes figures, tables, and a log. It does not reproduce the manuscript estimates.

## Run

From this directory:

```bash
pip install -r requirements.txt
python run_pipeline.py
```

The run rewrites `data/synthetic/` from `core/analysis.yaml`, then writes:

- `data/results/*.csv`
- `data/logs/run_*.log`

The figure notebooks use `scienceplots` Nature style and Arial. `visualizer/fig1.pdf` and `visualizer/fig2.pdf` are already in the repository. Redrawing them needs Arial on the machine; if that font is absent, Matplotlib substitutes another sans-serif.

Open `visualizer/01_fig1.ipynb` and `visualizer/02_fig2.ipynb` and run them. They read those files and write `visualizer/fig1.pdf` and `visualizer/fig2.pdf`. The notebooks in `analysis/` run the same estimators and print the tables. Start either notebook from this directory or from its own folder.

## Figure map

Fig. 1(c) matches AIGC-adopting and non-adopting creators. Fig. 1(d) matches interactions with platform-labeled AIGC and unlabeled content, with a 3 s duration caliper. Fig. 1(e) is the Scale-over-Engagement Index, SoEI_t = ln(S_t / E_t). Panels (a) and (b) of the manuscript figure are the platform description and the schematic. They are not drawn from this extract.

Fig. 2(a) matches platform-labeled AIGC and unlabeled videos on 31-day exposure. Fig. 2(b), left, is the differenced Granger test of s_t^a on M_t. Fig. 2(b), right, reports θ for the four exposure-tier shares w_{k,t}. Fig. 2(c) is the partial association of R_t with S_t and with E_t. The slope is the Newey-West coefficient.

**Fig. 2(d)** is θ for ln(e_t), the overall full-view rate and the overall valid-view rate, with respect to ln(s_t^a). The earlier creator-supply elasticity panel was removed from the manuscript and is not estimated here.

**Fig. 2(e)** is the share of exposures allocated to platform-labeled AIGC under population-feedback versus individual-feedback distribution. **Fig. 2(f)** is the individual-minus-population difference at 1, 2, 4, and 8 weeks.

## Symbols

| Column | Manuscript |
|---|---|
| `labeled_supply` | s_t^a, platform-labeled AIGC upload count |
| `unlabeled_supply` | s_t^u, unlabeled upload count |
| `first_day_median_exposure` | M_t |
| `rel_supply` | S_t = s_t^a / s_t^u |
| `rel_engagement` | E_t, ratio of mean valid-view rates |
| `rel_exposure` | R_t, ratio of 31-day median exposures |
| `share_le_10` … `share_above_1000` | w_{k,t} for ≤10, 11–100, 101–1,000, >1,000 |
| `log_full_view_rate`, `log_valid_view_rate` | ln(e_t) |
| `log_labeled_supply`, `log_unlabeled_supply` | ln(s_t^a), ln(s_t^u) |

SoEI_t = ln(S_t / E_t). The Granger null is γ_1 = … = γ_p = 0. The ARDL long-run multiplier is θ = −π / ρ.

## Methods

Matching (`core/matching.py`) blocks on categorical covariates, then takes the Euclidean nearest neighbor with replacement on z-scored continuous covariates. The four matching notebooks show that process: the exact cells, the pooled scale, who is dropped, and how often a comparison unit is reused. A caliper rejects neighbors outside a hard limit and is not part of the distance. Column lists live in `core/analysis.yaml`. Changing a list is how a second specification is run. There is no separate matcher per figure.

The mean contrast follows the manuscript orientation: AIGC-adopting minus non-adopting, platform-labeled AIGC minus unlabeled content, or population-feedback minus individual-feedback. Its variance (`core/imbens.py`) is the Abadie and Imbens (2006) variance for one-to-one matching with replacement. Up to five nearest neighbors in the same arm and the same exact cell estimate the conditional outcome variance. A comparison unit used K times has weight K squared. The Hodges-Lehmann median and Wilcoxon signed-rank p-value (`core/rank_tests.py`) are the rank summary reported beside that mean. Balance (`core/balance.py`) reports standardized mean differences, variance ratios, and a propensity-score overlap. The manuscript calls a continuous covariate balanced when the absolute standardized difference is below 0.1 and the variance ratio is between 0.5 and 2.

Granger tests (`core/granger.py`) are fit on first differences. The joint p-value is a residual moving-block bootstrap under the restricted model that drops the lags of platform-labeled AIGC supply. Blocks are 7 days.

Level OLS (`core/level_ols.py`) and ARDL (`core/ardl.py`) use Newey-West standard errors with a 7-day bandwidth. The ARDL long-run multiplier is θ = −π / ρ from the error-correction form. The manuscript uses Case 3 (unrestricted intercept, no trend) with two supply regressors. A level relationship is supported when F is at least 4.81 and the t statistic on ρ is at most -3.53. F below 3.80 is unsupported. The band from 3.80 up to 4.81 is inconclusive, and t does not resolve it. CUSUM and CUSUMSQ are the largest departure of the recursive residuals from their 5% bands; a ratio below 1 means the path stays inside.

The integration screen (`core/integration.py`) uses ADF and Phillips-Perron, both with an intercept, and KPSS if those two disagree at 5%.

## Sensitivity

| Check | Code | Manuscript |
|---|---|---|
| Omitted confounding | `analysis/robustness/matching/omitted.ipynb` | SI-2, Cinelli and Hazlett |
| Unlabeled-pool contamination | `analysis/robustness/matching/misclassification.ipynb` | SI-5, φ up to 0.474 |
| Granger lag, reverse order, unlabeled supply | `analysis/robustness/timeseries/granger.ipynb` | SI-4.2 |
| Covariate balance | `analysis/robustness/matching/balance.ipynb` | absolute SMD below 0.1; variance ratio in [0.5, 2] |
| ARDL bounds, t on rho, Breusch-Godfrey, CUSUM, CUSUMSQ, lag grid | `analysis/robustness/timeseries/ardl.ipynb` | SI-4.4 |
| Level OLS and the ARDL(2, 2) follow-up | `analysis/robustness/timeseries/ols.ipynb` | Newey-West, Breusch-Godfrey, Breusch-Pagan |
| Integration order | `analysis/robustness/timeseries/order.ipynb` | SI-4.1 |
| Duration caliper at 2, 3, and 5 seconds | `analysis/robustness/matching/caliper.ipynb` | Main text uses 3 seconds |

The contamination notebook uses the published detector calibration. It does not rerun the video audit.

## Layout

```text
analysis/01_matching_creator_production.ipynb
analysis/02_matching_consumer_engagement.ipynb
analysis/03_matching_video_exposure.ipynb
analysis/04_granger_causality.ipynb
analysis/05_ardl_long_run.ipynb
analysis/06_level_ols.ipynb
analysis/07_matching_algorithm_design.ipynb
analysis/robustness/matching/     omitted confounding, balance, misclassification, caliper
analysis/robustness/timeseries/   ARDL, Granger, level OLS, integration order
core/         estimators and core/analysis.yaml
visualizer/   Figure 1 and Figure 2 notebooks. They read data/ and write the pdf
data/         synthetic tables, result tables, logs, schema
```

## Data

Platform logs are proprietary and are not in this repository. `data/DATA.md` states which features of the synthetic sample follow the manuscript and which exist only so the code can run. Holiday and campaign flags are placeholders with the same role as the manuscript controls. They are not the platform calendar. Upload date is stored as `upload_week` so exact-match cells stay non-empty.

## License and citation

MIT, see `LICENSE`. To cite this repository, use `CITATION.cff`.
