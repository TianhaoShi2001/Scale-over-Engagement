# %% [markdown]
# # Figure 2
#
# Nature style, the 3×3 layout, CCDF, Granger bracket, tier forest,
# partial-residual panels, engagement elasticities, and the algorithm panels.
#
# Panel (d) is the consumer engagement elasticity. The creator-supply panel
# is not drawn.

# %%
import os
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scienceplots

warnings.filterwarnings("ignore")
os.environ["SEABORN_RC"] = "False"
plt.style.use("nature")

BASE_FONTSIZE = 14
TICK_FONTSIZE = BASE_FONTSIZE - 1
TITLE_FONTSIZE = BASE_FONTSIZE + 1
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": TICK_FONTSIZE,
    "figure.dpi": 150,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
})

HERE = Path.cwd().resolve()
for candidate in [HERE, *HERE.parents]:
    if (candidate / "run_pipeline.py").exists():
        sys.path.insert(0, str(candidate))
        ROOT = candidate
        break
else:
    raise FileNotFoundError("Start from the repository, or from visualizer/.")

RESULTS = ROOT / "data" / "results"
if not (RESULTS / "video_pairs.csv").exists():
    from core.pipeline import run_all

    run_all(ROOT)

videos = pd.read_csv(RESULTS / "video_pairs.csv")
granger = pd.read_csv(RESULTS / "granger_baseline.csv").sort_values("term")
tiers = pd.read_csv(RESULTS / "ardl_tiers.csv")
ols = pd.read_csv(RESULTS / "level_ols.csv").set_index("term")
partial = pd.read_csv(RESULTS / "level_ols_partial.csv")
engagement = pd.read_csv(RESULTS / "ardl_engagement.csv")
algorithm = pd.read_csv(RESULTS / "algorithm_pairs.csv")
effects = pd.read_csv(RESULTS / "algorithm_effects.csv").set_index("outcome")

# %%
fig = plt.figure(figsize=(22, 18), dpi=150)
gs = GridSpec(3, 3, figure=fig, hspace=0.30, wspace=0.35)
axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]

# %%
def get_ccdf(data_series):
    data_series = pd.Series(data_series).dropna()
    if data_series.empty or data_series.max() == 0:
        return np.array([0.0]), np.array([1.0])
    vc = data_series.value_counts().sort_index()
    return vc.index.to_numpy(float), 1 - (vc.to_numpy(float) / vc.sum()).cumsum()


for idx, metric, name in [
    (0, "exposure_31d", "Cumulative Exposure (Show Count, 31-day)"),
    (1, "days_to_90", "Days to Reach 90% of Cumulative Exposure (31-day)"),
]:
    ax = axes[0][idx]
    for label, suffix, col in [("Unlabeled", "_comparison", "#81b1e9"), ("AIGC", "_focal", "#88d4b7")]:
        x, y = get_ccdf(videos[f"{metric}{suffix}"])
        ax.plot(x, y, color=col, lw=3.0, label=label)
    ax.set_xlabel(name, fontsize=BASE_FONTSIZE + 1)
    ax.set_ylabel("CCDF", fontsize=BASE_FONTSIZE + 1)
    ax.legend(loc="upper right", frameon=False, fontsize=BASE_FONTSIZE)
    if idx == 0:
        ax.set_xscale("function", functions=(lambda x: np.log10(x + 1), lambda y: 10**y - 1))
        xmax = float(videos[["exposure_31d_focal", "exposure_31d_comparison"]].to_numpy().max())
        if xmax > 0:
            ticks = [0] + [10**i for i in range(1, int(np.log10(xmax)) + 2) if 10**i <= xmax * 1.2]
            ax.set_xticks(ticks)
            ax.set_xticklabels(["0"] + [f"$10^{{{int(np.log10(t))}}}$" for t in ticks[1:]])
        both = videos[["exposure_31d_focal", "exposure_31d_comparison"]].to_numpy(float)
        lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
        if lo > 1 and (hi + 1) / (lo + 1) < 12:
            pad = 0.12 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            inside = [tick for tick in ticks if lo - pad <= tick <= hi + pad]
            if len(inside) < 2:
                inside = [float(x) for x in np.linspace(lo, hi, 4)]
            ax.set_xticks(inside)
            labels = []
            for tick in inside:
                if tick >= 10 and abs(np.log10(tick) - round(np.log10(tick))) < 0.05:
                    labels.append(f"$10^{{{int(round(np.log10(tick)))}}}$")
                else:
                    labels.append(str(int(round(tick))))
            ax.set_xticklabels(labels)
    else:
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, 32.2)
        ax.set_xticks([1, 2, 4, 8, 16, 31])
        ax.set_xticklabels([1, 2, 4, 8, 16, 31])

# %%
ax_granger = axes[1][0]
lags = np.arange(1, len(granger) + 1)
coeffs = granger["estimate"].to_numpy(float)
yerr = 1.96 * granger["standard_error"].to_numpy(float)
ax_granger.errorbar(lags, coeffs, yerr=yerr, fmt="o", color="#88d4b7", ecolor="black", capsize=5)
ax_granger.scatter(lags, coeffs, color="#88d4b7", s=64, edgecolors="black", zorder=4)
ax_granger.axhline(0, color="black", linestyle="--", alpha=0.7)
ax_granger.set_xticks(lags)
ax_granger.set_xticklabels([f"Lag {int(lag)}" for lag in lags])
ax_granger.set_xlim(float(lags[0]) - 0.55, float(lags[-1]) + 0.55)


def draw_bracket(ax, x0, x1, y, h, lw=1.8):
    ax.plot([x0, x1], [y, y], color="black", lw=lw, solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0], [y, y - h], color="black", lw=lw, solid_capstyle="butt", zorder=5)
    ax.plot([x1, x1], [y, y - h], color="black", lw=lw, solid_capstyle="butt", zorder=5)


top = float(np.max(coeffs + yerr))
bot = float(np.min(coeffs - yerr))
span = max(top - bot, 1e-6)
h = 0.14 * span
feet = max(top, 0.0) + 0.08 * span
bracket_y = feet + h
draw_bracket(ax_granger, float(lags[0]) - 0.25, float(lags[-1]) + 0.25, bracket_y, h=h)
p_boot = float(granger["bootstrap_p"].iloc[0])
p_text = f"p = {p_boot:.3f}" if p_boot >= 0.001 else "p < 0.001"
ax_granger.text(
    float(np.mean(lags)),
    bracket_y + 0.06 * span,
    f"Granger {p_text}",
    ha="center", va="bottom", fontsize=TICK_FONTSIZE, color="black",
)
ax_granger.set_ylim(bot - 0.18 * span, bracket_y + 0.45 * span)
ax_granger.set_ylabel("Coeff. on $\\Delta$ First-Day Median Exposure\n(of New AIGC)")
ax_granger.set_xlabel(r"Lagged $\Delta$ AIGC Supply Scale")

ax_tiers = axes[1][1]
tier_order = [
    ("share_above_1000", "High\n(>1000)"),
    ("share_101_to_1000", "Medium\n-High\n(101–1000)"),
    ("share_11_to_100", "Medium\n-Low\n(11–100)"),
    ("share_le_10", "Low\n(≤10)"),
]
tier_rows = tiers[tiers["regressor"] == "labeled_supply"].set_index("outcome")
y_pos = np.arange(len(tier_order))
theta = np.array([tier_rows.loc[name, "theta"] for name, _ in tier_order], float)
theta_se = np.array([tier_rows.loc[name, "theta_standard_error"] for name, _ in tier_order], float)
ci_low = theta - 1.96 * theta_se
ci_high = theta + 1.96 * theta_se
xerr = np.array([theta - ci_low, ci_high - theta])
ax_tiers.errorbar(theta, y_pos, xerr=xerr, fmt="none", ecolor="black", capsize=5, alpha=0.9)
ax_tiers.scatter(theta, y_pos, color="#88d4b7", s=64, edgecolors="black", zorder=4)
ax_tiers.axvline(0, color="black", linestyle="--", alpha=0.7)
ax_tiers.set_yticks(y_pos)
ax_tiers.set_yticklabels([label for _, label in tier_order])
ax_tiers.invert_yaxis()
ax_tiers.set_ylabel("Exposure Tiers")
ax_tiers.set_xlabel(
    "Long-Run Coefficient of AIGC Supply Scale on\n Share of New AIGC in Different Exposure Tiers",
    fontsize=BASE_FONTSIZE,
    labelpad=10,
)
if np.nanmax(np.abs(theta)) < 1e-3:
    ax_tiers.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6), useMathText=False)

# %%
titles = [
    "Regression Controlling for Relative Consumer Engagement",
    "Regression Controlling for Relative Supply Scale",
]
xlabels = [
    "Relative Supply Scale of AIGC (to Unlabeled)",
    "Relative Consumer Engagement for AIGC (to Unlabeled)",
]
for j, regressor in enumerate(("rel_supply", "rel_engagement")):
    ax = axes[2][j]
    cloud = partial[partial["regressor"] == regressor]
    x = cloud["residual_regressor"].to_numpy(float)
    y = cloud["residual_outcome"].to_numpy(float)
    ax.scatter(x, y, s=11, alpha=0.4, color="#66c2a5")
    xline = np.linspace(x.min(), x.max(), 100)
    beta = float(ols.loc[regressor, "estimate"])
    ci = (float(ols.loc[regressor, "ci95_low"]), float(ols.loc[regressor, "ci95_high"]))
    ax.plot(xline, beta * xline, color="black", lw=2.0)
    ax.fill_between(xline, ci[0] * xline, ci[1] * xline, color="#ccece6", alpha=0.4)
    ax.set_title(titles[j], fontsize=BASE_FONTSIZE - 2)
    ax.set_xlabel(xlabels[j])
    ax.set_ylabel("Relative Median Exposure\n (of New AIGC to Unlabeled)")
    stats_text = f"β = {beta:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
    if j == 0:
        ax.text(
            0.97, 0.92, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=TICK_FONTSIZE - 1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, boxstyle="round,pad=0.4"),
        )
    else:
        ax.text(
            0.03, 0.92, stats_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=TICK_FONTSIZE - 1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, boxstyle="round,pad=0.4"),
        )
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")

# %%
ax = axes[0][2]
items = []
for outcome in ("log_full_view_rate", "log_valid_view_rate"):
    for regressor in ("log_labeled_supply", "log_unlabeled_supply"):
        row = engagement[(engagement["outcome"] == outcome) & (engagement["regressor"] == regressor)].iloc[0]
        items.append({"beta": float(row["theta"]), "ci": (float(row["theta_ci95_low"]), float(row["theta_ci95_high"]))})
# items: AIGC full, Unlabeled full, AIGC valid, Unlabeled valid
for g, offset in (("Unlabeled", 0.2), ("AIGC", -0.2)):
    idx = [1, 3] if g == "Unlabeled" else [0, 2]
    means = [items[i]["beta"] for i in idx]
    cis = [items[i]["ci"] for i in idx]
    yerr = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
    xpos = np.arange(2) + offset
    ax.bar(
        xpos, means, width=0.35,
        color="#81b1e9" if g == "Unlabeled" else "#88d4b7",
        edgecolor="black", label=g,
    )
    ax.errorbar(xpos, means, yerr=yerr, fmt="none", ecolor="black", capsize=3)
ax.set_xlabel("Long-Run Elasticities on\nConsumers' Engagement Depth")
ax.set_xticks(np.arange(2))
ax.set_xticklabels(["Overall Full-View", "Overall Valid-View"])
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
ax.axhline(0, color="black", alpha=0.6)
ax.set_ylabel("Long-Run Elasticity")

# %%
ax_f = axes[1][2]
for name, suffix, color in [
    ("Population Feedback", "_focal", "green"),
    ("Individual Feedback", "_comparison", "#88d4b7"),
]:
    series = algorithm[f"labeled_exposure_share{suffix}"].dropna().to_numpy(float)
    s_vals = np.sort(series)
    ax_f.plot(s_vals, 1 - np.arange(1, len(s_vals) + 1) / len(s_vals), color=color, lw=2.2, label=name)
ax_f.set_xlabel("Share of exposure to AIGC")
ax_f.set_ylabel("CCDF")
ax_f.legend(frameon=False)

# %%
weeks = ["share_week_1", "share_week_2", "share_week_4", "share_week_8"]
diff_vals = -effects.loc[weeks, "mean_difference"].to_numpy(float)
half = 1.96 * effects.loc[weeks, "standard_error"].to_numpy(float)
ci_lower = diff_vals - half
ci_upper = diff_vals + half
marker_color = "#88d4b7" if float(diff_vals[-1]) > 0 else "orange"
ax_g = axes[2][2]
x = np.arange(len(weeks))
ax_g.plot(x, diff_vals, linestyle="--", linewidth=1, alpha=0.25, color="black")
ax_g.scatter(x, diff_vals, color=marker_color, s=40, marker="o", zorder=3)
ax_g.errorbar(
    x, diff_vals,
    yerr=[diff_vals - ci_lower, ci_upper - diff_vals],
    fmt="o", color="gray", ecolor="black",
    capsize=6, elinewidth=1.5,
    markersize=8, markeredgecolor="black", zorder=1,
)
ax_g.set_xticks(x)
ax_g.set_xticklabels(["1w", "2w", "4w", "8w"], fontsize=TICK_FONTSIZE)
ax_g.set_xlabel("Evaluation Time Horizon", fontsize=BASE_FONTSIZE)
ax_g.set_ylabel(
    "Difference in share of exposure to AIGC\n" r"(Individual $-$ Population)",
    fontsize=BASE_FONTSIZE,
)
step = 0.005
lo = np.floor((float(ci_lower.min()) - step) / step) * step
hi = np.ceil((float(ci_upper.max()) + step) / step) * step
ax_g.set_yticks(np.round(np.arange(lo, hi + step / 2, step), 3))
ax_g.tick_params(axis="y", labelsize=TICK_FONTSIZE)
ax_g.grid(alpha=0)

# %%
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
gs.update(hspace=0.30, wspace=0.35)
for row in axes:
    for ax in row:
        pos = ax.get_position()
        height = pos.height * 0.92
        ax.set_position([pos.x0, pos.y1 - height, pos.width, height])


def add_label(ax_group, label):
    boxes = [ax.get_position() for ax in ax_group]
    fig.text(
        min(bb.x0 for bb in boxes) - 0.02,
        max(bb.y1 for bb in boxes) + 0.01,
        label,
        fontsize=22, fontweight="bold", fontfamily="Arial", ha="left", va="bottom",
    )


add_label([axes[0][0], axes[0][1]], "(a)")
add_label([axes[1][0], axes[1][1]], "(b)")
add_label([axes[2][0], axes[2][1]], "(c)")
add_label([axes[0][2]], "(d)")
add_label([axes[1][2]], "(e)")
add_label([axes[2][2]], "(f)")

fig.savefig(ROOT / "visualizer" / "fig2.pdf", dpi=1200, bbox_inches="tight", pad_inches=0.05)
fig.savefig(ROOT / "visualizer" / "fig2.png", dpi=120, bbox_inches="tight", pad_inches=0.05)
plt.show()
print("wrote visualizer/fig2.pdf")
