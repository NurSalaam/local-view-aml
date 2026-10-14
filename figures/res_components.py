"""Experiment 2's decomposition of the shortfall at HI-Medium's 107 qualifying
banks, for Chapter 4.

Data: Experiment 2, the `components` check on HI-Medium for the tree and GNN
families. Each Run's banks.csv holds one row per qualifying bank with the
per-bank components (tree: median over the eight configurations of the
per-configuration difference; GNN: mean over the three seeds); its
metrics.json holds the weighted mean of each component under Experiment 1's
weights 1/s_B^2 and the 95 % interval from the shared bootstrap draws
(`readings`). The rows are path 1 of the grid. Both Runs use the (network,
qualifying) scenario as fitted, not its variant with the count held equal.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "res_components.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
RED, GREY = "#c2352c", "#c3bfb6"

ROWS = (
    ("examples", "Value of training examples", "(ledger, qualifying) − (ledger, own)"),
    ("sight", "Value of sight", "(network, qualifying) − (ledger, qualifying)"),
    ("stub", "Extending the pool to every bank", "(network, all) − (network, qualifying)"),
)
FAMILIES = (("tree", "Tree family"), ("gnn", "GNN family"))



def reading(family):
    run = find(index("exp2"), check="components", dataset="HI-Medium", family=family)
    banks = pd.read_csv(run / "artifacts" / "banks.csv", dtype={"bank": str})
    readings = json.loads((run / "metrics.json").read_text())["readings"]
    return banks, readings


def panel(ax, banks, readings, label, title):
    jitter = np.random.default_rng(0).uniform(-0.26, 0.26, len(banks))
    for row, (component, _, _) in enumerate(ROWS):
        ax.scatter(banks[f"{component}_delta"], row + jitter, s=9, color=GREY,
                   edgecolor=MUTED, linewidth=0.4, zorder=2)
        lo, hi = readings[component]["ci95"]
        mean = readings[component]["weighted_mean"]
        ax.fill_betweenx([row - 0.38, row + 0.38], lo - 0.006, hi + 0.006, color="white",
                         linewidth=0, zorder=3)
        ax.fill_betweenx([row - 0.38, row + 0.38], lo, hi, color=RED, alpha=0.3,
                         linewidth=0, zorder=4)
        ax.plot([mean, mean], [row - 0.38, row + 0.38], color=RED, linewidth=1.1, zorder=5)
    ax.axvline(0.0, color=INK, linewidth=0.7, zorder=1)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([])
    for row, (_, name, cells) in enumerate(ROWS):
        ax.text(-0.02, row - 0.14, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=10, color=INK)
        ax.text(-0.02, row + 0.18, cells, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=10, color=MUTED)
    for row in range(1, len(ROWS)):
        ax.axhline(row - 0.5, color=RULE, linewidth=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(MUTED)
    ax.text(0.012, 1.03, f"{label}  {title}", ha="left", va="bottom",
            transform=mpl.transforms.blended_transform_factory(ax.figure.transFigure, ax.transAxes),
            fontsize=10, color=INK)


figure, axes = plt.subplots(2, 1, figsize=(6.3, 5.6), sharex=True)
for ax, label, (family, title) in zip(axes, ("(a)", "(b)"), FAMILIES):
    panel(ax, *reading(family), label, title)
axes[1].set_xlabel("Difference in average precision", fontsize=10, color=INK)
axes[1].set_xlim(-0.56, 0.9)
axes[1].set_xticks(np.arange(-0.5, 0.81, 0.25))

handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=3.5, markerfacecolor=GREY,
           markeredgecolor=MUTED, markeredgewidth=0.4, label="One qualifying bank"),
    (Patch(facecolor=RED, alpha=0.3, linewidth=0), Line2D([], [], color=RED, linewidth=1.1)),
]
figure.legend(handles, ["One qualifying bank", "Weighted mean and its 95 % interval"],
              loc="lower center", ncol=2, frameon=False, fontsize=10,
              bbox_to_anchor=(0.6, 0.005), handlelength=1.6, columnspacing=2.0)
figure.subplots_adjust(left=0.42, right=0.98, top=0.952, bottom=0.155, hspace=0.25)
figure.savefig(OUT)
