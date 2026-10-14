"""The gain from borrowing and the shortfall at each bank of HI-Medium, for Chapter 6.

Data: Experiment 4, the `policy` checks on HI-Medium for the tree and
GNN families. Each check's `banks.csv` gives per bank its shortfall from
Experiment 1 (`delta_exp1`), the gain of the detector it ends with over its own
(`policy_gain`, zero where it keeps its own detector), whether it borrows
(`borrow`) and its quarter by confirmed cases (`quartile`). Runs are found by
scanning Experiment 4's Run log. The weighted mean gain and the weighted mean
shortfall over the first quarter are recomputed with the weights 1/s_B^2 and
asserted against the recorded `primary.weighted_mean` and
`threshold.weighted_mean_shortfall`; their ratio is the share of the shortfall
recovered that the caption quotes.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
RUNS = index("exp4")
OUT = ROOT / "figures" / "res_borrowed.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
BLUE, RED = "#2a5d9f", "#c2352c"
TEXT_SIZE = 10.1
FAMILIES = (("tree", "(a) Tree family", BLUE), ("gnn", "(b) GNN family", RED))

cells = {}
for family, _, _ in FAMILIES:
    path = find(RUNS, check="policy", dataset="HI-Medium", family=family)
    banks = pd.read_csv(path / "artifacts" / "banks.csv", dtype={"bank": str})
    metrics = json.loads((path / "metrics.json").read_text())
    first = banks[banks.quartile == 1]
    weights = 1.0 / first.weight_se_exp1 ** 2
    gain = float((weights * first.policy_gain).sum() / weights.sum())
    shortfall = float((weights * first.delta_exp1).sum() / weights.sum())
    assert abs(gain - metrics["primary"]["weighted_mean"]) < 1e-9, (family, gain)
    assert abs(shortfall - metrics["threshold"]["weighted_mean_shortfall"]) < 1e-9, \
        (family, shortfall)
    cells[family] = (path.name, banks, gain / shortfall)

low = min(min(b.delta_exp1.min(), b.policy_gain.min()) for _, b, _ in cells.values()) - 0.05
high = max(max(b.delta_exp1.max(), b.policy_gain.max()) for _, b, _ in cells.values()) + 0.05

figure, axes = plt.subplots(1, 2, figsize=(6.3, 4.1), sharex=True, sharey=True,
                            gridspec_kw=dict(left=0.105, right=0.985, bottom=0.215, top=0.94,
                                             wspace=0.07))
for ax, (family, title, colour) in zip(axes, FAMILIES):
    _, banks, _ = cells[family]
    first = banks.quartile == 1
    ax.plot([low, high], [low, high], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.scatter(banks.delta_exp1[~first], banks.policy_gain[~first], s=14, marker="o",
               facecolor="none", edgecolor="#9a958c", lw=0.6, zorder=2)
    ax.scatter(banks.delta_exp1[first], banks.policy_gain[first], s=18, marker="o",
               facecolor=colour, edgecolor=colour, lw=0.6, zorder=3)
    ax.set_title(title, fontsize=TEXT_SIZE, color=INK, loc="left")
    ax.set_xlabel("Shortfall", fontsize=TEXT_SIZE, color=INK)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=TEXT_SIZE, color=MUTED, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
axes[0].set_ylabel("Gain from borrowing", fontsize=TEXT_SIZE, color=INK)

handles = [
    Line2D([], [], ls="none", marker="o", markersize=4.6, markerfacecolor=BLUE,
           markeredgecolor=BLUE, label="Tree family, 4 to 22 confirmed cases"),
    Line2D([], [], ls="none", marker="o", markersize=4.6, markerfacecolor=RED,
           markeredgecolor=RED, label="GNN family, 4 to 22 confirmed cases"),
    Line2D([], [], ls="none", marker="o", markersize=4.2, markerfacecolor="none",
           markeredgecolor="#9a958c", label="More than 22 confirmed cases"),
    Line2D([], [], color=MUTED, lw=0.8, ls="--", label="Gain equal to the shortfall"),
]
figure.legend(handles=handles, loc="lower center", ncol=2, fontsize=TEXT_SIZE, frameon=False,
              bbox_to_anchor=(0.53, 0.0), handletextpad=0.4, columnspacing=1.4)

figure.savefig(OUT)
print("saved", OUT)
for family, _, _ in FAMILIES:
    name, banks, share = cells[family]
    first = banks[banks.quartile == 1]
    print(family, name, "banks", len(banks), "first quarter", len(first),
          "keep their own detector", int((~banks["borrow"]).sum()),
          "gain below zero", int((banks.policy_gain < 0).sum()),
          f"share of the shortfall recovered {share:.2f}")
