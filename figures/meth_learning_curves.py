"""Learning curves of the GNN family on HI-Medium, for Chapter 3.

Data: the epoch trace each GNN fit records (`trace` in its metrics.json): per
epoch, the mean training loss over the epoch's batches (`loss`, the weighted
binary cross-entropy the optimiser minimises) and the validation average
precision (`val_ap`). Three panels: the three seeds of the detector with the
global view, (network, all), Experiment 1's `gnn_global` Runs; and the
(ledger, own) and (network, own) fits of the qualifying bank at the lower
median of training-window payments (`train_rows` of the roster of the tree
family's `shortfall` Run, ties by identifier), Experiment 1's `gnn_solo` and
Experiment 2's `gnn_nown` Runs. Validation loss is not recorded. The filled
marker is the epoch whose state is kept, the first with the highest
validation AP. Runs are found by config in the Run logs.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from m.runlog import find, index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "meth_learning_curves.pdf"
KEY = "HI-Medium"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "font.size": 10,
})

INK, MUTED = "#1c1c1c", "#6b6b6b"
SEEDS = ((0, "#2a5d9f", "-"), (1, "#c2352c", "--"), (2, "#6b6b6b", ":"))


def trace(path, scenario, seed):
    metrics = json.loads((path / "metrics.json").read_text())
    assert "val_loss" not in metrics["trace"][0]
    return pd.DataFrame([{"scenario": scenario, "seed": seed, "epoch": t["epoch"] + 1,
                          "train_loss": t["loss"], "val_ap": t["val_ap"]}
                         for t in metrics["trace"]])


RUNS1, RUNS2 = index("exp1"), index("exp2")
shortfall = find(RUNS1, check="shortfall", dataset_key=KEY, family="tree")
roster = pd.read_csv(shortfall / "artifacts" / "roster.csv", dtype={"bank": str})
ordered = roster.sort_values(["train_rows", "bank"])
bank = ordered["bank"].iloc[(len(ordered) - 1) // 2]
fits = {"network,all": lambda s: find(RUNS1, check="gnn_global", dataset_key=KEY, seed=s),
        "ledger,own": lambda s: find(RUNS1, check="gnn_solo", dataset_key=KEY,
                                     bank=bank, seed=s),
        "network,own": lambda s: find(RUNS2, check="gnn_nown", dataset_key=KEY,
                                      bank=bank, seed=s)}
curves = pd.concat([trace(fit(seed), scenario, seed)
                    for scenario, fit in fits.items() for seed, *_ in SEEDS],
                   ignore_index=True)
panels = (
    ("network,all", "(a) (network, all)\nthe whole of HI-Medium"),
    ("ledger,own", f"(b) (ledger, own)\nbank {bank}"),
    ("network,own", f"(c) (network, own)\nbank {bank}"),
)

figure, axes = plt.subplots(2, 3, figsize=(6.3, 4.4), sharex=True,
                            gridspec_kw=dict(left=0.1, right=0.985, bottom=0.17,
                                             top=0.89, wspace=0.32, hspace=0.12))
for column, (scenario, title) in enumerate(panels):
    loss_ax, ap_ax = axes[0, column], axes[1, column]
    loss_ax.set_title(title, fontsize=10, color=INK, linespacing=1.3)
    for seed, colour, style in SEEDS:
        c = curves[(curves.scenario == scenario) & (curves.seed == seed)]
        best = c.loc[c.val_ap.idxmax()]
        loss_ax.plot(c.epoch, c.train_loss, color=colour, ls=style, lw=1.1)
        ap_ax.plot(c.epoch, c.val_ap, color=colour, ls=style, lw=1.1)
        ap_ax.plot(best.epoch, best.val_ap, "o", ms=4, color=colour)
        ap_ax.plot(c.epoch.iloc[-1], c.val_ap.iloc[-1], "x", ms=4, mew=1, color=colour)
    loss_ax.set_yscale("log")
    loss_ax.set_ylim(min(0.01, curves[curves.scenario == scenario].train_loss.min()), None)
    loss_ax.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    loss_ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    loss_ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ap_ax.set_ylim(0, None)
    ap_ax.set_xlabel("Epoch", fontsize=10, color=INK)
    for ax in (loss_ax, ap_ax):
        ax.set_xlim(0.5, 15.5)
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([1, 5, 10, 15]))
        ax.tick_params(labelsize=9, color=MUTED, length=3)
        ax.tick_params(which="minor", color=MUTED, length=1.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.6)
axes[0, 0].set_ylabel("Training loss\n(log scale)", fontsize=10, color=INK)
axes[1, 0].set_ylabel("Validation\naverage precision", fontsize=10, color=INK)
handles = [Line2D([], [], color=colour, ls=style, lw=1.1, label=f"Seed {seed}")
           for seed, colour, style in SEEDS]
handles += [Line2D([], [], ls="none", marker="o", ms=4, color=MUTED, label="Kept epoch"),
            Line2D([], [], ls="none", marker="x", ms=4, mew=1, color=MUTED, label="Last epoch")]
figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.0), ncol=5,
              fontsize=9, frameon=False, handletextpad=0.4, columnspacing=1.2)
figure.savefig(OUT)
print("saved", OUT, "| bank", bank)
for scenario, _ in panels:
    for seed, *_ in SEEDS:
        c = curves[(curves.scenario == scenario) & (curves.seed == seed)]
        best = c.loc[c.val_ap.idxmax()]
        print(f"{scenario} seed {seed}: epochs {len(c)}, kept epoch {int(best.epoch)}, "
              f"val AP {best.val_ap:.3f}, last {c.val_ap.iloc[-1]:.3f}, "
              f"loss {c.train_loss.iloc[0]:.4f} -> {c.train_loss.iloc[-1]:.4f}")
