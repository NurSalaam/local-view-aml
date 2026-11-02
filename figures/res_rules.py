"""The difference between the out-of-fold rank correlation of each of the fifteen
label-free regression models and that of the scale model on HI-Medium, for Chapter 5.

Data: Experiment 3, the `uncertainty` check of `scripts/exp3_readings.py`, which
reproduces the recorded `selection` Runs' models. Per family and model, `delta` is the
difference of the mean over ten repeats of the out-of-fold Spearman correlation
from that of the scale model, and `q05` and `q95` are the 5th and 95th
percentiles of that difference over 1,000 paired redraws of the 107 banks, the
out-of-fold predictions held as recorded. A point marks the difference and a
bar runs from the 5th to the 95th percentile; the vertical line marks no
difference.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "res_rules.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED = "#1c1c1c", "#6b6b6b"
BLUE = "#2a5d9f"
FAMILIES = (("tree", "(a) Tree family"), ("gnn", "(b) GNN family"))

GROUPS = {
    "openness": "Openness",
    "knowledge": "Counterparty knowledge",
    "spread": "Banks of the counterparties",
    "structure": "Internal structure",
    "mix": "Mix",
}
ADDED = {
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
MODELS = {**GROUPS, **ADDED}


run = find(index("exp3"), check="uncertainty", dataset_key="HI-Medium")
metrics = json.loads((run / "metrics.json").read_text())
cells = {family: [(MODELS[m], metrics[family]["banks"][m]["delta"],
                   metrics[family]["banks"][m]["q05"], metrics[family]["banks"][m]["q95"])
                  for m in MODELS]
         for family, _ in FAMILIES}

gap = 0.8
positions = [i if i < len(GROUPS) else i + gap for i in range(len(MODELS))]

figure, axes = plt.subplots(1, 2, figsize=(6.3, 4.4), sharey=True,
                            gridspec_kw=dict(left=0.35, right=0.985, bottom=0.12,
                                             top=0.95, wspace=0.08))
for ax, (family, title) in zip(axes, FAMILIES):
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    for y, (_, delta, q05, q95) in zip(positions, cells[family]):
        ax.plot([q05, q95], [y, y], color=BLUE, lw=1.2, solid_capstyle="butt", zorder=2)
        ax.scatter(delta, y, s=16, color=BLUE, zorder=3)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlim(-0.95, 0.15)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.1f}".replace("-", "−")))
    ax.tick_params(labelsize=10, color=MUTED, length=3)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_xlabel("Difference in out-of-fold ρ\nfrom the scale model", fontsize=10, color=INK)
axes[0].set_yticks(positions)
axes[0].set_yticklabels([name for name, *_ in cells["tree"]])
axes[0].set_ylim(positions[-1] + 0.8, -0.8)
figure.savefig(OUT)
print("saved", OUT, "from", run.name)
for family, _ in FAMILIES:
    print(f"{family}: scale {metrics[family]['scale_score']:.4f}")
    for name, delta, q05, q95 in cells[family]:
        print(f"  {name:38s} {delta:+.3f} [{q05:+.3f}, {q95:+.3f}]")
