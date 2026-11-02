"""The shortfall and the bank's size on HI-Medium as generated, for Chapter 4.

Data: Experiment 1, the `shortfall` checks on HI-Medium for the tree
and GNN families. Each check's `banks.csv` gives per bank the shortfall
Delta_B (`delta`) and the bank's test-window payments (`test_rows`, payments
with the bank at either end). The Spearman rho printed in the panel is computed
here, between the shortfall and the natural log of size. Runs are found by
scanning Experiment 1's Run log.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
RUNS = index("exp1")
OUT = ROOT / "figures" / "res_size.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
BLUE, RED = "#2a5d9f", "#c2352c"
FAMILIES = (("tree", "Tree family"), ("gnn", "GNN family"))
STYLE = {
    "tree": dict(marker="o", s=16, facecolor=BLUE, edgecolor=BLUE, lw=0.6),
    "gnn": dict(marker="^", s=20, facecolor="none", edgecolor=RED, lw=0.9),
}



def signed(value, places):
    return f"{value:+.{places}f}".replace("-", "−")


cells = {}
for family, _ in FAMILIES:
    observed_dir = find(RUNS, check="shortfall", dataset="HI-Medium", family=family)
    banks = pd.read_csv(observed_dir / "artifacts" / "banks.csv", dtype={"bank": str})
    rho = spearmanr(banks.delta, np.log(banks.test_rows)).statistic
    cells[family] = (observed_dir.name, banks, rho)

all_banks = [banks for _, banks, _ in cells.values()]
y_low = min(b.delta.min() for b in all_banks)
y_high = max(b.delta.max() for b in all_banks)
x_low = min(b.test_rows.min() for b in all_banks)
x_high = max(b.test_rows.max() for b in all_banks)

figure, ax = plt.subplots(figsize=(6.3, 3.4),
                          gridspec_kw=dict(left=0.105, right=0.985, bottom=0.255, top=0.975))
ax.axhline(0, color=INK, lw=0.8, zorder=1)
lines = []
for family, name in FAMILIES:
    _, banks, rho = cells[family]
    ax.scatter(banks.test_rows, banks.delta, zorder=3 if family == "tree" else 4,
               **STYLE[family])
    lines.append(f"{name}: {len(banks)} banks, Spearman ρ {signed(rho, 2)}")
ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=10,
        color=MUTED, ha="right", va="top", linespacing=1.35)
ax.set_xscale("log")
ax.set_xlim(x_low / 1.25, x_high * 1.25)
ax.set_ylim(y_low - 0.07, y_high + 0.12)
ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([2e3, 5e3, 1e4, 2e4, 5e4]))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: f"{v:.1f}".replace("-", "−")))
ax.tick_params(labelsize=10, color=MUTED, length=3)
ax.tick_params(which="minor", color=MUTED, length=1.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
    ax.spines[side].set_linewidth(0.6)
ax.set_ylabel("Shortfall, average precision", fontsize=10, color=INK)
ax.set_xlabel("Size, the test-window payments of the bank (log scale)", fontsize=10, color=INK)
figure.legend(handles=[
    Line2D([], [], ls="none", marker="o", markersize=4.5, markerfacecolor=BLUE,
           markeredgecolor=BLUE, label="Tree family"),
    Line2D([], [], ls="none", marker="^", markersize=5, markerfacecolor="none",
           markeredgecolor=RED, markeredgewidth=0.9, label="GNN family"),
], loc="lower center", bbox_to_anchor=(0.54, 0.0), ncol=2, fontsize=10, frameon=False,
    handletextpad=0.3, columnspacing=1.6)
figure.savefig(OUT)
print("saved", OUT)
for family, name in FAMILIES:
    run, banks, rho = cells[family]
    print(f"{name}: run {run}, k {len(banks)}, rho {rho:+.4f}, delta {banks.delta.min():+.4f} "
          f"to {banks.delta.max():+.4f}, test_rows {banks.test_rows.min()} to "
          f"{banks.test_rows.max()}")
