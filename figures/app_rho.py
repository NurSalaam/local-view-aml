"""How alike the two families rank HI-Medium's banks, for the appendix to Chapter 4.

Data: Experiment 1, the `shortfall` checks on HI-Medium for the tree
and GNN families. Each bank is placed at its rank by the tree family's
shortfall (`delta` in `banks.csv`) and its rank by the GNN family's; rank 1
is the largest shortfall. The rank correlation printed in the panel is the
one the verdict Run recorded (`criteria.2.rho` of the `verdict` check with
`criterion_3_rule` "triple"); the script asserts that it reproduces from the
two `banks.csv` files.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
RUNS = index("exp1")
OUT = ROOT / "figures" / "app_rho.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
GREY = "#c3bfb6"



def banks_of(family):
    directory = find(RUNS, check="shortfall", dataset="HI-Medium", family=family)
    return pd.read_csv(directory / "artifacts" / "banks.csv", dtype={"bank": str}).set_index("bank")


tree, gnn = banks_of("tree"), banks_of("gnn")
assert list(tree.index) == list(gnn.index)
verdict = find(RUNS, check="verdict", criterion_3_rule="triple")
rho = json.loads((verdict / "metrics.json").read_text())["criteria"]["2"]["rho"]
recomputed = spearmanr(tree["delta"], gnn["delta"]).statistic
assert abs(rho - recomputed) < 1e-9, (rho, recomputed)

tree_rank = tree["delta"].rank(ascending=False)
gnn_rank = gnn["delta"].rank(ascending=False)
k = len(tree)

figure, ax = plt.subplots(figsize=(3.6, 3.5),
                          gridspec_kw=dict(left=0.19, right=0.97, bottom=0.15, top=0.97))
ax.plot([1, k], [1, k], color=RULE, linewidth=0.8, zorder=1)
ax.scatter(tree_rank, gnn_rank, s=12, color=GREY, edgecolor=MUTED, linewidth=0.5, zorder=2)
ax.text(0.04, 0.96, f"{k} banks\nSpearman $\\rho$ = {rho:.3f}", transform=ax.transAxes,
        fontsize=9.6, color=MUTED, ha="left", va="top", linespacing=1.35)
ax.set_xlim(0, k + 1)
ax.set_ylim(0, k + 1)
ax.set_xticks([1, 25, 50, 75, 100])
ax.set_yticks([1, 25, 50, 75, 100])
ax.set_xlabel("Rank by the tree family's shortfall", fontsize=10, color=INK)
ax.set_ylabel("Rank by the GNN family's shortfall", fontsize=10, color=INK)
ax.tick_params(labelsize=9.6, colors=INK, length=3)
ax.set_aspect("equal")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
    ax.spines[side].set_linewidth(0.6)
figure.savefig(OUT)
print("saved", OUT, "| verdict", verdict.name, "| rho", rho, "recomputed", recomputed)
