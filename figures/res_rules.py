"""The out-of-fold rank correlation of the sixteen rules on properties and of the rule on
scale and confirmed cases on HI-Medium, for Chapter 5.

Data: Experiment 3. The descriptor selection's `selection` Runs give the
sixteen label-free candidates per family (`candidates.csv`: `score`, the mean
over ten repeats of the out-of-fold Spearman correlation, and
`paired_q05_vs_baseline`, the 5th percentile of the paired bank-redraw
difference from the scale rule). Experiment 3's `ceiling` Runs give the ten label-using
candidates the same way (`score`, `q05_vs_baseline`). The scale rule appears in
both sets with the same score, which the script asserts, and is drawn once.
A rule is drawn filled when its paired 5th percentile exceeds zero.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "res_rules.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
BLUE, RED = "#2a5d9f", "#c2352c"
FAMILIES = (("tree", "(a) Tree family"), ("gnn", "(b) GNN family"))

LABEL_FREE = {
    "scale": "Scale",
    "openness": "Openness",
    "knowledge": "Counterparty knowledge",
    "spread": "Banks of the counterparties",
    "structure": "Internal structure",
    "mix": "Mix",
    "scale+internal_share": "Scale + internal share",
    "scale+boundary_share": "Scale + boundary share",
    "scale+external_recurrence": "Scale + recurring counterparties",
    "scale+external_encounters": "Scale + encounters per counterparty",
    "scale+external_banks": "Scale + counterparty banks",
    "scale+external_hhi": "Scale + concentration",
    "scale+activity_gini": "Scale + activity inequality",
    "scale+reciprocity": "Scale + reciprocity",
    "scale+format_entropy": "Scale + format entropy",
    "scale+currency_entropy": "Scale + currency entropy",
}
LABEL_USING = {
    "scale+laundering_count": "Scale + confirmed cases",
}



def rows(family):
    free_dir = find(index("exp3"), check="selection", dataset_key="HI-Medium", family=family)
    using_dir = find(index("exp3"), check="ceiling", dataset_key="HI-Medium", family=family)
    free = pd.read_csv(free_dir / "artifacts" / "candidates.csv").set_index("candidate")
    using = pd.read_csv(using_dir / "artifacts" / "candidates.csv").set_index("candidate")
    assert abs(free.loc["scale", "score"] - using.loc["scale", "score"]) < 1e-12
    out = [(LABEL_FREE[c], free.loc[c, "score"], free.loc[c, "paired_q05_vs_baseline"] > 0,
            "free") for c in LABEL_FREE]
    out += [(LABEL_USING[c], using.loc[c, "score"], using.loc[c, "q05_vs_baseline"] > 0,
             "using") for c in LABEL_USING]
    return free_dir.name, using_dir.name, out, free.loc["scale", "score"]


cells = {family: rows(family) for family, _ in FAMILIES}
n = len(LABEL_FREE) + len(LABEL_USING)
gap = 0.8
positions = [i if i < len(LABEL_FREE) else i + gap for i in range(n)]

figure, axes = plt.subplots(1, 2, figsize=(6.3, 4.8), sharey=True,
                            gridspec_kw=dict(left=0.35, right=0.985, bottom=0.19,
                                             top=0.955, wspace=0.08))
for ax, (family, title) in zip(axes, FAMILIES):
    _, _, entries, scale = cells[family]
    ax.axvline(scale, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    for y, (_, score, beats, kind) in zip(positions, entries):
        colour = BLUE if kind == "free" else RED
        ax.plot([0, score], [y, y], color=RULE, lw=0.6, zorder=1)
        ax.scatter(score, y, s=22, zorder=3, edgecolor=colour, lw=0.9,
                   facecolor=colour if beats else "white",
                   marker="o" if kind == "free" else "D")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlim(-0.12, 0.95)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v:.1f}".replace("-", "−")))
    ax.tick_params(labelsize=10, color=MUTED, length=3)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_xlabel("Out-of-fold ρ", fontsize=10, color=INK)
axes[0].set_yticks(positions)
axes[0].set_yticklabels([name for name, *_ in cells["tree"][2]])
axes[0].set_ylim(positions[-1] + 0.8, -0.8)
for label, y in zip(axes[0].get_yticklabels(), positions):
    label.set_color(INK if y < len(LABEL_FREE) else RED)
figure.legend(handles=[
    Line2D([], [], ls="none", marker="o", markersize=4.5, markerfacecolor="white",
           markeredgecolor=BLUE, label="Rule on properties"),
    Line2D([], [], ls="none", marker="D", markersize=4.2, markerfacecolor="white",
           markeredgecolor=RED, label="Rule on scale and confirmed cases"),
    Line2D([], [], ls="none", marker="D", markersize=4.2, markerfacecolor=RED,
           markeredgecolor=RED, label="Filled: Outperforms the scale rule"),
    Line2D([], [], color=MUTED, lw=0.8, ls="--", label="The scale rule"),
], loc="lower right", bbox_to_anchor=(0.985, 0.0), ncol=2, fontsize=10, frameon=False,
    handletextpad=0.3, columnspacing=1.0)
figure.savefig(OUT)
print("saved", OUT)
for family, _ in FAMILIES:
    free_run, using_run, entries, scale = cells[family]
    print(f"{family}: selection {free_run}, ceiling {using_run}, scale {scale:.4f}")
    for name, score, beats, kind in entries:
        print(f"  {kind:5s} {name:38s} {score:+.4f} {'outperforms' if beats else ''}")
