"""Column names shared by the generator, the estimators, and the figures.

Table names stay in words. The manuscript symbols are:

    s_t^a  labeled_supply          platform-labeled AIGC upload count
    s_t^u  unlabeled_supply        unlabeled-content upload count
    M_t    first_day_median_exposure
    S_t    rel_supply              s_t^a / s_t^u
    E_t    rel_engagement          ratio of mean valid-view rates
    R_t    rel_exposure            ratio of 31-day median exposures
    w_k,t  share_*                 exposure-tier share
    ln(e_t) log_full_view_rate, log_valid_view_rate
    ln(s_t^a), ln(s_t^u)  log_labeled_supply, log_unlabeled_supply
    SoEI_t = ln(S_t / E_t)
"""

# Pair columns are "{covariate}_focal" and "{covariate}_comparison".
FOCAL = "_focal"
COMPARISON = "_comparison"

# s_t^a and s_t^u in the Granger and ARDL equations.
LABELED_SUPPLY = "labeled_supply"
UNLABELED_SUPPLY = "unlabeled_supply"
LOG_LABELED_SUPPLY = "log_labeled_supply"
LOG_UNLABELED_SUPPLY = "log_unlabeled_supply"

# M_t, first-day median exposure of platform-labeled AIGC videos uploaded on day t.
FIRST_DAY_MEDIAN_EXPOSURE = "first_day_median_exposure"

# ln(e_t): overall full-view rate and overall valid-view rate.
LOG_FULL_VIEW_RATE = "log_full_view_rate"
LOG_VALID_VIEW_RATE = "log_valid_view_rate"

EXPOSURE_SHARES = (
    "share_le_10",
    "share_11_to_100",
    "share_101_to_1000",
    "share_above_1000",
)
