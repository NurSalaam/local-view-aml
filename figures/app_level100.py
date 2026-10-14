"""The score with 100 confirmed cases against the range at the full count, for the appendix to Chapter 4.

Data: Experiment 2, the `dial` check on HI-Medium for the tree and GNN
families (metrics.json, per bank). Only the 13 banks holding more than 100
confirmed cases have a level of 100. For each, the dot is the mean over the
five draws of the score with 100 confirmed cases (tree: each draw's median
over the eight configurations; GNN: seed 0), and the bar is the range of the
bank's score at the full count (`band_all`: over the eight configurations or
the three seeds). The horizontal tick on each bar is the full-count point the
dot is compared with (`ap_all`): the median over the eight configurations for
the tree family, seed 0 for the GNN family. Banks are ordered by their full
count; the dashed line separates the banks holding 103 to 122 confirmed cases from those holding 132
to 265. A dot counts as within the range when band floor <= mean <= band
ceiling, as the Experiment 2 facts Run counts it.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "app_level100.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
RED, GREY = "#c2352c", "#c3bfb6"
FAMILIES = (("tree", "(a) Tree family"), ("gnn", "(b) GNN family"))
SPLIT = 125



def level_100(family):
    run = find(index("exp2"), check="dial", dataset="HI-Medium", family=family)
    banks = json.loads((run / "metrics.json").read_text())["banks"]
    rows = sorted(((b["train_pos"], bank, b["by_level"]["100"]["mean"], b["band_all"], b["ap_all"])
                   for bank, b in banks.items() if "100" in b["by_level"]))
    return run.name, rows


figure, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), sharey=True,
                            gridspec_kw=dict(left=0.09, right=0.985, bottom=0.32, top=0.935,
                                             wspace=0.08))
for ax, (family, title) in zip(axes, FAMILIES):
    run, rows = level_100(family)
    within = 0
    for x, (count, bank, mean, (lo, hi), full) in enumerate(rows):
        inside = lo <= mean <= hi
        within += inside
        ax.plot([x, x], [lo, hi], color=MUTED, linewidth=3.2, solid_capstyle="butt", zorder=2)
        ax.plot([x - 0.3, x + 0.3], [full, full], color=INK, linewidth=1.1, zorder=2.5)
        ax.plot(x, mean, marker="o", markersize=4.2, zorder=3,
                markerfacecolor=INK if inside else "white",
                markeredgecolor=INK if inside else RED, markeredgewidth=1.0)
    split = sum(count < SPLIT for count, *_ in rows) - 0.5
    ax.axvline(split, color=MUTED, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(count) for count, *_ in rows], fontsize=10, rotation=90)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0.3, 1.0)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    ax.set_xlabel("Bank, by its full count of confirmed cases", fontsize=10, color=INK)
    ax.grid(axis="y", color=RULE, linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=10, colors=INK, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    print(f"{title}: run {run}, banks {len(rows)}, within {within}, "
          f"counts {[count for count, *_ in rows]}")
axes[0].set_ylabel("Average precision", fontsize=10, color=INK)
figure.legend(handles=[
    Line2D([], [], color=MUTED, linewidth=3.2, label="Range at the full count"),
    Line2D([], [], color=INK, linewidth=1.1, label="Score at the full count"),
    Line2D([], [], ls="none", marker="o", markersize=4.2, markerfacecolor=INK,
           markeredgecolor=INK, label="100 confirmed cases, within the range"),
    Line2D([], [], ls="none", marker="o", markersize=4.2, markerfacecolor="white",
           markeredgecolor=RED, markeredgewidth=1.0, label="100 confirmed cases, below it"),
], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=10, frameon=False,
    handletextpad=0.5, columnspacing=1.6, handlelength=1.6, labelspacing=0.3)
figure.savefig(OUT)
print("saved", OUT)
