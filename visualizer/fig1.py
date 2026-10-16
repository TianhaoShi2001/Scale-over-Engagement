# %% [markdown]
# # Figure 1
#
# `scienceplots` Nature style, Arial, the two colors, CCDF on a log1p axis,
# grouped means with 95% intervals, boxplot, and the SoE scatter.
#
# Panels (c), (d) and (e) are the analytical panels. Panels (a) and (b) in the
# manuscript are the platform description and the schematic.

# %%
import os
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scienceplots

warnings.filterwarnings("ignore")
os.environ["SEABORN_RC"] = "False"
plt.style.use("nature")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path.cwd().resolve()
for candidate in [HERE, *HERE.parents]:
    if (candidate / "run_pipeline.py").exists():
        sys.path.insert(0, str(candidate))
        ROOT = candidate
        break
else:
    raise FileNotFoundError("Start from the repository, or from visualizer/.")

RESULTS = ROOT / "data" / "results"
SYNTHETIC = ROOT / "data" / "synthetic"
if not (RESULTS / "creator_pairs.csv").exists():
    from core.pipeline import run_all

    run_all(ROOT)

# %%
FIGSIZE = (24, 14)
FONT_SIZE = 23
LINE_WIDTH = 2.5
COLOR_NON = "#81b1e9"
COLOR_AI = "#88d4b7"
COLOR_HGC = "#81b1e9"
COLOR_AIGC = "#88d4b7"

creators = pd.read_csv(RESULTS / "creator_pairs.csv")
consumers = pd.read_csv(RESULTS / "consumer_pairs.csv")
daily = pd.read_csv(SYNTHETIC / "daily_panel.csv")


def get_ccdf(data_series):
    data_series = pd.Series(data_series).dropna()
    if data_series.empty or data_series.max() == 0:
        return np.array([0.0]), np.array([1.0])
    value_counts = data_series.value_counts()
    value_counts_df = pd.DataFrame({"value": value_counts.index, "count": value_counts.values})
    value_counts_df = value_counts_df.sort_values("value")
    value_counts_df["count"] = value_counts_df["count"] / value_counts_df["count"].sum()
    x = value_counts_df["value"].to_numpy(float)
    y = (1 - value_counts_df["count"].cumsum()).to_numpy(float)
    if x[0] > 0:
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 1)
    return x, y


def arm_mean_ci(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, 1.96 * se

# %%
fig = plt.figure(figsize=FIGSIZE)
gs_main = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25, height_ratios=[1, 1], width_ratios=[1, 1, 1])

creator_panels = [
    ("video_volume", "Creation Volume"),
    ("total_valid_views", "Valid-View Count"),
    ("total_full_views", "Full-View Count"),
]
creator_axes = [fig.add_subplot(gs_main[0, i]) for i in range(3)]
for ax, (outcome, group_name) in zip(creator_axes, creator_panels):
    data_dict = {
        "Non-adopting Creator": creators[f"{outcome}_comparison"],
        "AIGC-adopting Creator": creators[f"{outcome}_focal"],
    }
    drawn = []
    for label, series in data_dict.items():
        x, y = get_ccdf(series)
        drawn.append(x)
        ax.plot(
            x, y,
            color=(COLOR_NON if label == "Non-adopting Creator" else COLOR_AI),
            linewidth=LINE_WIDTH,
            label=label,
        )
    ax.set_xscale("function", functions=(lambda x: np.log10(x + 1), lambda y: 10**y - 1))
    x_max = max(float(np.max(x)) for x in drawn) if drawn else 1
    ticks = [0] + [10**k for k in range(1, int(np.log10(x_max + 1)) + 2)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [rf"$10^{int(np.log10(t))}$" for t in ticks[1:]])
    both = np.concatenate([series.to_numpy(float) for series in data_dict.values()])
    lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
    if lo > 1 and (hi + 1) / (lo + 1) < 12:
        pad = 0.12 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        inside = [tick for tick in ticks if lo - pad <= tick <= hi + pad]
        if len(inside) >= 2:
            ax.set_xticks(inside)
            ax.set_xticklabels(["0" if tick == 0 else rf"$10^{int(np.log10(tick))}$" for tick in inside])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(group_name, fontsize=FONT_SIZE)
    ax.set_ylabel("CCDF", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE - 4)

gs_bottom = gs_main[1, :].subgridspec(1, 2, width_ratios=[1, 0.72], wspace=0.2)
gs_left = gs_bottom[0].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=0.3)
ax_bar = fig.add_subplot(gs_left[0])
ax_box = fig.add_subplot(gs_left[1])

plot_metrics = [
    ("early_retention", "Early-Retention"),
    ("valid_view_rate", "Valid-View"),
    ("full_view_rate", "Full-View"),
]
x_idx = np.arange(len(plot_metrics))
bar_width, group_gap = 0.35, 0.05
offsets = {"Unlabeled": -(bar_width / 2 + group_gap / 2), "AIGC": +(bar_width / 2 + group_gap / 2)}
fig2_linewidth = 1.2
for group, suffix in (("Unlabeled", "_comparison"), ("AIGC", "_focal")):
    means, cis = [], []
    for outcome, _ in plot_metrics:
        mean, ci = arm_mean_ci(consumers[f"{outcome}{suffix}"])
        means.append(mean)
        cis.append(ci)
    xpos = x_idx + offsets[group]
    ax_bar.bar(
        xpos, means, width=bar_width, label=group,
        color=(COLOR_HGC if group == "Unlabeled" else COLOR_AIGC),
        edgecolor="black", linewidth=fig2_linewidth,
    )
    ax_bar.errorbar(xpos, means, yerr=cis, fmt="none", ecolor="black", elinewidth=fig2_linewidth, capsize=3)
ax_bar.set_xticks(x_idx)
ax_bar.set_xticklabels([label for _, label in plot_metrics], fontsize=FONT_SIZE - 1)
ax_bar.tick_params(labelsize=FONT_SIZE - 1, axis="y")
ax_bar.tick_params(labelsize=FONT_SIZE, axis="x", rotation=12)
ax_bar.set_ylabel("Rate", fontsize=FONT_SIZE)
ax_bar.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE - 4)
ax_bar.grid(True, alpha=0, axis="y")

bplot = ax_box.boxplot(
    [consumers["view_fraction_comparison"], consumers["view_fraction_focal"]],
    positions=[1, 2], widths=0.55, patch_artist=True,
    showmeans=True, meanline=False, showfliers=False,
    meanprops={"marker": "^", "markeredgecolor": "black", "markerfacecolor": "white", "markersize": 7},
    whiskerprops=dict(color="black", linewidth=fig2_linewidth),
    capprops=dict(color="black", linewidth=fig2_linewidth),
    medianprops=dict(color="black", linewidth=fig2_linewidth),
)
bplot["boxes"][0].set_facecolor(COLOR_HGC)
bplot["boxes"][1].set_facecolor(COLOR_AIGC)
for box in bplot["boxes"]:
    box.set_edgecolor("black")
whisk_top = max(float(w.get_ydata()[1]) for w in bplot["whiskers"])
ax_box.set_ylabel("Normalized View Fraction", fontsize=FONT_SIZE)
ax_box.set_ylim(0, min(1.05, whisk_top * 1.15))
ax_box.set_xticks([1, 2])
ax_box.set_xticklabels(["Unlabeled", "AIGC"], fontsize=FONT_SIZE)
ax_box.tick_params(axis="x", length=0, labelsize=FONT_SIZE, pad=15)
ax_box.tick_params(axis="y", labelsize=FONT_SIZE - 1, direction="out")
ax_box.grid(True, alpha=0, axis="y")

ax_sop = fig.add_subplot(gs_bottom[1])
supply = daily["rel_supply"].to_numpy(float)
engagement = daily["rel_engagement"].to_numpy(float)
sop = np.log(supply / engagement)
vmin, vmax = -0.5, 1
cmap = plt.get_cmap("Reds")
scatter = ax_sop.scatter(engagement, supply, c=sop, cmap=cmap, s=35, alpha=0.8, vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax_sop)
cax = divider.append_axes("right", size="3%", pad=0.15)
cb = plt.colorbar(scatter, cax=cax)
cb.set_label("SoE Index", fontsize=FONT_SIZE)
cb.ax.tick_params(labelsize=FONT_SIZE - 1)
cb.ax.set_yticks([-0.5, 0.0, 0.5, 1.0])

min_x, max_x = float(engagement.min()), float(engagement.max())
x_line = np.linspace(min_x * 0.95, max_x * 1.05, 500)
if max_x < 1:
    x_line = x_line[x_line < 0.999]


def sop_curve(x, level):
    return np.exp(level) * x


for rate, label_text in ((1, "SoEI = 0\n(S = E)"), (1.5, "SoEI = ln(1.5)\n(S = 1.5E)")):
    level = np.log(rate)
    color = cmap((level - vmin) / (vmax - vmin))
    y_line = np.array([sop_curve(xx, level) if xx > 0 else np.nan for xx in x_line])
    valid = ~np.isnan(y_line)
    if np.any(valid):
        ax_sop.plot(x_line[valid], y_line[valid], linestyle="--", linewidth=1.4, alpha=0.5, color=color)
        ids = np.where(valid)[0]
        last_idx = ids[int(0.82 * (len(ids) - 1))]
        ax_sop.annotate(
            label_text,
            xy=(x_line[last_idx], y_line[last_idx]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=FONT_SIZE - 4,
            alpha=0.5,
        )

ax_sop.set_xlabel(r"Relative Consumer Engagement with AIGC (E)", fontsize=FONT_SIZE)
ax_sop.set_ylabel(r"Relative AIGC Supply Scale (S)", fontsize=FONT_SIZE)
ax_sop.set_title("SoE Index", fontsize=FONT_SIZE, pad=15)
ax_sop.tick_params(labelsize=FONT_SIZE - 1)

plt.tight_layout()
sop_pos = ax_sop.get_position()
square_size = min(sop_pos.width, sop_pos.height)
ax_sop.set_position([
    sop_pos.x1 - square_size,
    sop_pos.y0 + (sop_pos.height - square_size) / 2,
    square_size,
    square_size,
])


def add_label(ax_group, label):
    boxes = [ax.get_position() for ax in ax_group]
    fig.text(
        min(bb.x0 for bb in boxes) - 0.03,
        max(bb.y1 for bb in boxes) + 0.012,
        label,
        fontsize=FONT_SIZE + 6,
        fontweight="bold",
        fontfamily="Arial",
        ha="left",
        va="bottom",
    )


add_label(creator_axes, "(c)")
add_label([ax_bar, ax_box], "(d)")
add_label([ax_sop], "(e)")

fig.savefig(ROOT / "visualizer" / "fig1.pdf", dpi=1200, bbox_inches="tight", pad_inches=0.05)
fig.savefig(ROOT / "visualizer" / "fig1.png", dpi=160, bbox_inches="tight", pad_inches=0.05)
plt.show()
print("wrote visualizer/fig1.pdf")
