"""The null check of Cochran's Q on two readings, for the appendix to Chapter 4.

Data: Experiment 1, the `shortfall` checks for HI-Medium's tree
family and for the GNN family on `sweden_100K_easy` (the AMLgentex dataset).
Each check's `banks.csv` gives per bank the shortfall (`delta`) and its error
bar (`se`), and for the GNN family the variance of its per-seed shortfalls
(`seed_var`); `replicates.npy` holds the 1,000 resampled shortfalls of every
bank. The null is rebuilt as `m.stats.null_q_check` builds it: every bank's
resampled shortfalls less its measured shortfall, plus, for the GNN family, a
normal draw of variance `seed_var / 3` from `default_rng(0)`; Q is computed on
each resample with the weights 1/se^2. The script asserts that the rebuilt
95th percentile and the observed Q reproduce the values the Run recorded.
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
RUNS = index("exp1")
OUT = ROOT / "figures" / "app_null.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
RED, GREY = "#c2352c", "#c3bfb6"
TEXT_PT = 10



def q_of(deltas, weights):
    mean = (deltas * weights).sum(axis=-1, keepdims=True) / weights.sum()
    return ((deltas - mean) ** 2 * weights).sum(axis=-1)


def null_reading(dataset, family):
    directory = find(RUNS, check="shortfall", dataset=dataset, family=family)
    banks = pd.read_csv(directory / "artifacts" / "banks.csv", dtype={"bank": str})
    reps = np.load(directory / "artifacts" / "replicates.npy").astype(np.float64)
    deltas, ses = banks["delta"].to_numpy(), banks["se"].to_numpy()
    centred = reps - deltas[None, :]
    if family == "gnn":
        rng = np.random.default_rng(0)
        centred = centred + rng.standard_normal(centred.shape) * np.sqrt(
            banks["seed_var"].to_numpy() / 3.0)[None, :]
    weights = 1.0 / ses ** 2
    null_q = q_of(centred, weights)
    observed = float(q_of(deltas, weights))
    recorded = json.loads((directory / "metrics.json").read_text())["primary"]
    q95 = float(np.quantile(null_q, 0.95))
    assert abs(q95 - recorded["null_check"]["null_q95"]) < 1e-3, (q95, recorded["null_check"])
    assert abs(observed - recorded["Q"]) < 1e-6, (observed, recorded["Q"])
    return directory.name, len(banks), null_q, recorded["null_check"]["null_q95"], recorded["Q"]


PANELS = (("HI-Medium", "tree", "(a) HI-Medium, tree family", 25),
          ("sweden_100K_easy", "gnn", "(b) AMLgentex dataset, GNN family", 5))

figure, axes = plt.subplots(1, 2, figsize=(6.3, 3.2),
                            gridspec_kw=dict(left=0.095, right=0.985, bottom=0.28, top=0.9,
                                             wspace=0.24))
for ax, (dataset, family, title, x_step) in zip(axes, PANELS):
    run, k, null_q, q95, observed = null_reading(dataset, family)
    high = max(null_q.max(), observed) if observed < 3 * q95 else null_q.max()
    bins = np.linspace(0, high * 1.02, 40)
    counts, _, _ = ax.hist(null_q, bins=bins, color=GREY, edgecolor=MUTED, linewidth=0.4,
                           zorder=2)
    top = counts.max() * 1.3
    ax.set_ylim(0, top)
    ax.axvline(q95, color=INK, linestyle="--", linewidth=1.0, zorder=3)
    ax.axvline(k - 1, color=MUTED, linestyle=":", linewidth=1.0, zorder=3)
    if observed <= bins[-1]:
        ax.axvline(observed, color=RED, linewidth=1.4, zorder=4)
    else:
        note = ax.text(bins[-1] * 0.02, top * 0.93,
                       f"observed $Q$ = {observed:,.1f},\nfar to the right",
                       fontsize=TEXT_PT, color=RED, ha="left", va="center")
        figure.canvas.draw()
        note_right = ax.transData.inverted().transform(
            note.get_window_extent().get_points())[1, 0]
        ax.annotate("", xy=(bins[-1], top * 0.93), xytext=(note_right + bins[-1] * 0.03, top * 0.93),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlim(0, bins[-1])
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.set_title(title, loc="left", fontsize=TEXT_PT, color=INK, pad=6)
    ax.set_xlabel("$Q$ on a resample with no real spread", fontsize=TEXT_PT, color=INK)
    ax.tick_params(labelsize=TEXT_PT, colors=INK, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    print(f"{title}: run {run}, k {k}, null q95 {q95:.2f}, observed Q {observed:.2f}, "
          f"null mean {null_q.mean():.2f}, null max {null_q.max():.2f}")
axes[0].set_ylabel("Resamples", fontsize=TEXT_PT, color=INK)
figure.legend(handles=[
    Patch(facecolor=GREY, edgecolor=MUTED, linewidth=0.4, label="$Q$ on 1,000 resamples"),
    Line2D([], [], color=INK, linestyle="--", linewidth=1.0, label="95th percentile"),
    Line2D([], [], color=MUTED, linestyle=":", linewidth=1.0, label="$k - 1$"),
    Line2D([], [], color=RED, linewidth=1.4, label="Observed $Q$"),
], loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=4, fontsize=TEXT_PT, frameon=False,
    handletextpad=0.4, columnspacing=1.4)
figure.savefig(OUT)
print("saved", OUT)
