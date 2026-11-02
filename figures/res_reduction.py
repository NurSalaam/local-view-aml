"""Experiment 2's reduction of confirmed cases at 20 qualifying banks, for Chapter 4.

Data: Experiment 2, the `dial` check on HI-Medium for the tree and GNN
families (metrics.json, per bank). At 5, 10, 30 and 100 confirmed cases a
bank's score is the mean over the five nested draws of its test average
precision; for the tree family each draw's value is the median over the
eight configurations. At the bank's full count the score is the one the dial
Run adopts from Experiment 1's solo Runs: the median over the eight
configurations (tree) or seed 0 (GNN, the seed the reduced fits use). A level
at or above a bank's full count is not fitted, so a curve ends at the bank's
own count. Every bank is drawn alike.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "res_reduction.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "font.size": 10,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
GREY = "#a19d94"
FONT = 10
FAMILIES = (("tree", "Tree family"), ("gnn", "GNN family"))



def curves(family):
    run = find(index("exp2"), check="dial", dataset="HI-Medium", family=family)
    banks = json.loads((run / "metrics.json").read_text())["banks"]
    return {bank: ([*b["levels"], b["train_pos"]],
                   [*(b["by_level"][str(level)]["mean"] for level in b["levels"]), b["ap_all"]])
            for bank, b in banks.items()}


def panel(ax, bank_curves, label, title):
    for counts, scores in bank_curves.values():
        ax.plot(counts, scores, color=GREY, linewidth=0.9, zorder=2)
        ax.plot(counts[-1], scores[-1], marker="o", markersize=2.6, color=INK, zorder=3)
    full = max(counts[-1] for counts, _ in bank_curves.values())
    ax.set_xscale("log")
    ax.set_xlim(4.2, 330)
    ax.set_xticks([5, 10, 30, 100, full])
    ax.set_xticks([], minor=True)
    ax.set_xticklabels(["5", "10", "30", "100", str(full)])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=RULE, linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=FONT, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.set_title(f"{label}  {title}", loc="left", fontsize=FONT, color=INK, pad=6)
    ax.set_xlabel("Confirmed cases among the training examples\n(log scale)", fontsize=FONT, color=INK)


figure, axes = plt.subplots(1, 2, figsize=(6.3, 4.0), sharey=True)
for ax, label, (family, title) in zip(axes, ("(a)", "(b)"), FAMILIES):
    panel(ax, curves(family), label, title)
axes[0].set_ylabel("Average precision on the test set of the bank", fontsize=FONT, color=INK)

handles = [
    Line2D([], [], color=GREY, linewidth=0.9, label="One of the 20 banks"),
    Line2D([], [], color=INK, linestyle="none", marker="o", markersize=2.6,
           label="Score at the full count of the bank"),
]
figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=FONT,
              bbox_to_anchor=(0.53, 0.0), handlelength=2.0, columnspacing=3.0)
figure.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.26, wspace=0.08)
figure.savefig(OUT)
