"""Every HI-Medium bank's shortfall with its interval, one panel per family, for Chapter 4.

Data: Experiment 1, the `shortfall` checks on HI-Medium for the tree
and GNN families. Each check's `banks.csv` holds, per qualifying
bank, the shortfall Delta_B (`delta`), its 95 % percentile interval from the
joint bootstrap of the test window (`ci_lo`, `ci_hi`) and its error bar s_B
(`se`). The dashed line is the weighted mean with weights 1/s_B^2, the check's
recorded `primary.weighted_mean`. Runs are found by scanning Experiment 1's Run log.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
RUNS = index("exp1")
OUT = ROOT / "figures" / "res_shortfalls.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "font.size": 10,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
BLUE, RED, GREY = "#2a5d9f", "#c2352c", "#c3bfb6"

FAMILIES = (
    ("tree", "Tree family", BLUE, "o"),
    ("gnn", "GNN family", RED, "^"),
)



def reading(family):
    directory = find(RUNS, check="shortfall", dataset="HI-Medium", family=family)
    banks = pd.read_csv(directory / "artifacts" / "banks.csv", dtype={"bank": str})
    metrics = json.loads((directory / "metrics.json").read_text())
    return directory.name, banks.sort_values("delta").reset_index(drop=True), metrics["primary"]


readings = {family: reading(family) for family, *_ in FAMILIES}
low = min(banks.ci_lo.min() for _, banks, _ in readings.values())
high = max(banks.ci_hi.max() for _, banks, _ in readings.values())
x_limits = (low - 0.05, high + 0.05)

figure, axes = plt.subplots(1, 2, figsize=(6.3, 4.5), sharex=True, sharey=True,
                            gridspec_kw=dict(left=0.085, right=0.985, bottom=0.11, top=0.93,
                                             wspace=0.08))
for ax, (family, name, colour, marker), letter in zip(axes, FAMILIES, "ab"):
    _, banks, primary = readings[family]
    rank = np.arange(1, len(banks) + 1)
    ax.hlines(rank, banks.ci_lo, banks.ci_hi, color=GREY, lw=0.8, zorder=1)
    ax.scatter(banks.delta, rank, s=10, marker=marker, color=colour, lw=0, zorder=3)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    mean = primary["weighted_mean"]
    ax.axvline(mean, color=INK, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
    ax.text(mean + 0.03, 3, f"Weighted\nmean {mean:+.3f}".replace("-", "−"),
            fontsize=10, color=INK, ha="left", va="bottom", linespacing=1.15,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"),
            zorder=4)
    ax.text(0.0, 1.015, f"({letter}) {name}", transform=ax.transAxes, fontsize=10,
            color=INK, ha="left", va="bottom")
    ax.set_xlim(*x_limits)
    ax.set_ylim(0, len(banks) + 1)
    ax.set_xticks([-0.4, 0.0, 0.4, 0.8])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v:.1f}".replace("-", "−")))
    ax.tick_params(labelsize=10, color=MUTED, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
axes[0].set_yticks([1, 25, 50, 75, 107])
axes[0].set_ylabel("Bank, ranked by shortfall (1 = smallest)", fontsize=10, color=INK)
figure.supxlabel("Shortfall, average precision", fontsize=10, color=INK, x=0.535, y=0.012)
figure.savefig(OUT)
print("saved", OUT)

for family, name, *_ in FAMILIES:
    run, banks, primary = readings[family]
    w = 1 / banks.se ** 2
    print(f"{name}: run {run}, k {len(banks)}, weighted mean {primary['weighted_mean']:+.4f} "
          f"(recomputed {(w * banks.delta).sum() / w.sum():+.4f}), "
          f"delta {banks.delta.min():+.4f} to {banks.delta.max():+.4f}, "
          f"median {banks.delta.median():+.4f}, "
          f"interval {banks.ci_lo.min():+.4f} to {banks.ci_hi.max():+.4f}, "
          f"delta < 0: {int((banks.delta < 0).sum())}, "
          f"interval wholly above 0: {int((banks.ci_lo > 0).sum())}, "
          f"wholly below 0: {int((banks.ci_hi < 0).sum())}, "
          f"covering 0: {int(((banks.ci_lo <= 0) & (banks.ci_hi >= 0)).sum())}")
print("x limits", x_limits)
