"""The four scenarios over one real neighbourhood of HI-Small, for Chapter 3.

The neighbourhood is found, not chosen: the laundering payments among the
eleven qualifying banks are split into connected components, the smallest
component that spans exactly four qualifying banks is the anchor, and every
account that transacts with one of its accounts is drawn beside it. Bank 023 is
the focal bank.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
LEDGERS = ROOT / "data" / "interim" / "exp1" / "ledgers"
OUT = ROOT / "figures" / "meth_scenarios.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
LAUNDERING, SEEN, UNSEEN = "#c2352c", "#2a5d9f", "#c3bfb6"
POOL_FILL, IDLE_FILL = "#fae7c9", "#f4f3f0"
FOCAL = "023"


def bank_of(node):
    return node.split("/")[0]


def hi_small_network():
    banks = pd.read_csv(LEDGERS / "HI-Small" / "roster.csv", dtype={"bank": str})["bank"]
    parts = []
    for bank in banks:
        part = pd.read_parquet(LEDGERS / "HI-Small" / f"{bank}.parquet",
                               columns=["ts", "from_bank", "from_acct", "to_bank",
                                        "to_acct", "is_laundering", "payment"])
        for column in ("from_bank", "to_bank", "from_acct", "to_acct"):
            part[column] = part[column].astype(str)
        parts.append(part)
    net = pd.concat(parts).drop_duplicates("payment").reset_index(drop=True)
    net["payer"] = net.from_bank + "/" + net.from_acct
    net["payee"] = net.to_bank + "/" + net.to_acct
    return list(banks), net[net.payer != net.payee].reset_index(drop=True)


roster_banks, network = hi_small_network()
laundering = network[network.is_laundering == 1]
components = nx.connected_components(nx.from_pandas_edgelist(laundering, "payer", "payee"))
chain = min((c for c in components
             if len({bank_of(n) for n in c}) == 4 and {bank_of(n) for n in c} <= set(roster_banks)),
            key=lambda c: (len(c), sorted(c)[0]))
assert FOCAL in {bank_of(n) for n in chain}

hood = set(chain)
for node in chain:
    touching = network[(network.payer == node) | (network.payee == node)]
    hood |= {n for n in set(touching.payer) | set(touching.payee) if bank_of(n) in roster_banks}
drawn = network[network.payer.isin(hood) & network.payee.isin(hood)].copy()
drawn["pair"] = [tuple(sorted(p)) for p in zip(drawn.payer, drawn.payee)]
pairs = drawn.groupby("pair").agg(payments=("payment", "size"),
                                  laundering=("is_laundering", "max")).reset_index()

others = sorted({bank_of(n) for n in hood} - {FOCAL})
weight = {b: {} for b in others}
for (u, v), n in zip(pairs.pair, pairs.payments):
    a, b = bank_of(u), bank_of(v)
    if a in weight and b in weight and a != b:
        weight[a][b] = weight[a].get(b, 0) + n
        weight[b][a] = weight[b].get(a, 0) + n
ring, remaining = [others[0]], set(others[1:])
while remaining:
    ring.append(max(remaining, key=lambda b: (weight[ring[-1]].get(b, 0), b)))
    remaining.discard(ring[-1])

centre = {FOCAL: np.array([0.0, 0.0])}
for i, bank in enumerate(ring):
    angle = np.pi / 2 + 2 * np.pi * i / len(ring)
    centre[bank] = 1.08 * np.array([np.cos(angle), np.sin(angle)])
place, disc = {}, {}
for bank in [FOCAL] + ring:
    members = sorted(n for n in hood if bank_of(n) == bank)
    spread = 0.0 if len(members) == 1 else 0.16
    disc[bank] = 0.30 if len(members) > 1 else 0.15
    facing = np.pi / 2 if bank == FOCAL else np.arctan2(-centre[bank][1], -centre[bank][0])
    for j, node in enumerate(members):
        angle = facing + 2 * np.pi * j / len(members)
        place[node] = centre[bank] + spread * np.array([np.cos(angle), np.sin(angle)])

focal_accounts = {n for n in hood if bank_of(n) == FOCAL}
visible = [bank_of(u) == FOCAL or bank_of(v) == FOCAL for u, v in pairs.pair]
visible_nodes = set(focal_accounts)
for (u, v), seen in zip(pairs.pair, visible):
    if seen:
        visible_nodes |= {u, v}
seen_pairs = int(np.sum(visible))
seen_payments = int(pairs.payments[visible].sum())
seen_laundering = int(pairs.laundering[visible].sum())
all_pairs, all_payments, all_laundering = len(pairs), int(pairs.payments.sum()), int(pairs.laundering.sum())


def panel(ax, sight, pool, title, footnote, bordered):
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-2.22, 1.62)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK if bordered else "#cdc9c1")
        spine.set_linewidth(1.4 if bordered else 0.6)
    for bank in [FOCAL] + ring:
        pooled = pool == "all" or bank == FOCAL
        ax.add_patch(Circle(centre[bank], disc[bank], zorder=0, lw=0.6, edgecolor=RULE,
                            facecolor=POOL_FILL if pooled else IDLE_FILL))
        # The payments between the focal bank and the bank above it would cross a label below.
        beside = bank == ring[0]
        ax.text(*(centre[bank] + ((disc[bank] + 0.05, 0.0) if beside else (0.0, -disc[bank] - 0.07))),
                bank, ha="left" if beside else "center", va="center" if beside else "top",
                fontsize=10, color=INK if bank == FOCAL else MUTED,
                fontweight="bold" if bank == FOCAL else "normal")
    widest = np.log1p(pairs.payments.max())
    for (u, v), n, is_laundering, seen in zip(pairs.pair, pairs.payments, pairs.laundering, visible):
        shown = sight == "network" or seen
        width = 0.6 + 1.7 * np.log1p(n) / widest
        ax.add_patch(FancyArrowPatch(
            place[u], place[v], connectionstyle="arc3,rad=0.13", arrowstyle="-",
            lw=width if shown else 0.6, zorder=2 if shown else 1,
            color=(LAUNDERING if is_laundering else SEEN) if shown else UNSEEN,
            linestyle="-" if shown else (0, (1.5, 1.8))))
    for node in hood:
        shown = sight == "network" or node in visible_nodes
        ax.scatter(*place[node], s=40 if node in focal_accounts else 24, zorder=3, lw=0.8,
                   facecolor=INK if node in focal_accounts else ("white" if shown else "#f7f6f3"),
                   edgecolor=INK if shown else UNSEEN)
    ax.text(-1.56, 1.56, title, fontsize=11, color=INK, va="top")
    ax.text(0.0, -2.16, footnote, fontsize=10, color=MUTED, ha="center", va="bottom")


figure = plt.figure(figsize=(6.3, 7.08))
grid = figure.add_gridspec(2, 2, left=0.10, right=0.976, bottom=0.1726, top=0.9926,
                           wspace=0.0517, hspace=0.049)
whole = (f"In sight: all {all_pairs} account pairs,\nall {all_payments} payments,\n"
         f"all {all_laundering} laundering payments")
partial = (f"In sight: {seen_pairs} of the {all_pairs} account pairs,\n{seen_payments} of the "
           f"{all_payments} payments,\n{seen_laundering} of the {all_laundering} laundering payments")
layout = [
    ((0, 0), "ledger", "all", r"$M(\mathrm{ledger},\,\mathrm{all})$", partial, False),
    ((0, 1), "network", "all", r"$M(\mathrm{network},\,\mathrm{all})$", whole, True),
    ((1, 0), "ledger", "own", r"$M(\mathrm{ledger},\,\mathrm{own})$", partial, True),
    ((1, 1), "network", "own", r"$M(\mathrm{network},\,\mathrm{own})$", whole, False),
]
for position, sight, pool, title, footnote, bordered in layout:
    ax = figure.add_subplot(grid[position])
    ax.set_anchor("E" if position[1] == 0 else "W")
    panel(ax, sight, pool, title, footnote, bordered)

figure.text(0.538, 0.1107, r"Sight $S$: the payments from which features and messages are computed",
            ha="center", fontsize=10, color=INK)
figure.text(0.3373, 0.1432, f"Bank {FOCAL}'s ledger", ha="center", fontsize=10, color=MUTED)
figure.text(0.7386, 0.1432, "The whole network", ha="center", fontsize=10, color=MUTED)
figure.text(0.0836, 0.5826, r"Training pool $P$: whose confirmed cases the detector learns from",
            va="center", rotation=90, ha="center", fontsize=10, color=INK)
figure.text(0.1195, 0.7925, "Every bank's", va="center", rotation=90, ha="center", fontsize=10, color=MUTED)
figure.text(0.1195, 0.3727, f"Bank {FOCAL}'s own", va="center", rotation=90, ha="center", fontsize=10, color=MUTED)
figure.legend(handles=[
    Line2D([], [], color=SEEN, lw=2.0, label="Payment within the detector's sight"),
    Line2D([], [], color=UNSEEN, lw=1.0, ls=(0, (1.5, 1.8)), label="Payment outside its sight"),
    Line2D([], [], color=LAUNDERING, lw=2.0, label="Laundering payment"),
    Patch(facecolor=POOL_FILL, edgecolor=RULE, label="Bank whose confirmed cases are in the pool"),
    Line2D([], [], marker="o", color="none", markerfacecolor=INK, markeredgecolor=INK,
           markersize=5.5, label=f"Account at bank {FOCAL}"),
], loc="lower center", bbox_to_anchor=(0.538, 0.0), ncol=2, handlelength=2.2,
    columnspacing=1.4, fontsize=10, frameon=False, borderaxespad=0.2)
figure.savefig(OUT)

print("saved", OUT)
print(f"{len(hood)} accounts at {len(ring) + 1} banks, {all_pairs} account pairs, "
      f"{len(drawn)} payments, {int(drawn.is_laundering.sum())} of them laundering")
print(f"bank {FOCAL} sees {seen_pairs} pairs, {seen_payments} payments, {seen_laundering} laundering")
print("the anchor chain, in time order:")
for _, row in laundering[laundering.payer.isin(chain) & laundering.payee.isin(chain)].sort_values("ts").iterrows():
    print(f"  {row.payer}  ->  {row.payee}")
print("banks in ring:", ring)
