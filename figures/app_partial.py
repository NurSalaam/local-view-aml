"""A six-bank example of the partial rank correlation, for the appendix to Chapter 4.

Constructed ranks, not data: six banks ranked by size (1 to 6), by confirmed
cases (1, 3, 2, 4, 6, 5) and by shortfall (6, 5, 3, 4, 2, 1). Panels (a) and
(b) fit a least-squares line of each rank on the size rank; panel (c) plots
the two sets of leftovers (the ranks less their lines) against each other.
The raw rank correlation, the two lines, the leftovers and the partial rank
correlation are printed so that the appendix text can be checked against
them.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "app_partial.pdf"

TEXT = 10.05

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "font.size": TEXT,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
RED, GREY = "#c2352c", "#c3bfb6"

size = np.arange(1, 7, dtype=float)
cases = np.array([1, 3, 2, 4, 6, 5], dtype=float)
shortfall = np.array([6, 5, 3, 4, 2, 1], dtype=float)


def line(y):
    slope, intercept = np.polyfit(size, y, 1)
    return intercept, slope, y - (intercept + slope * size)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


raw = pearson(cases, shortfall)
cases_line, shortfall_line = line(cases), line(shortfall)
partial = pearson(cases_line[2], shortfall_line[2])

figure, axes = plt.subplots(1, 3, figsize=(6.3, 2.7),
                            gridspec_kw=dict(left=0.068, right=0.98, bottom=0.18, top=0.895,
                                             wspace=0.45))
for ax, y, (a, b, _), title, ylabel in (
        (axes[0], cases, cases_line, "(a) Confirmed cases on size", "Rank by confirmed cases"),
        (axes[1], shortfall, shortfall_line, "(b) Shortfall on size", "Rank by shortfall")):
    ax.plot([0.6, 6.4], [a + b * 0.6, a + b * 6.4], color=RED, linewidth=1.1, zorder=2)
    for x, value in zip(size, y):
        ax.plot([x, x], [value, a + b * x], color=MUTED, linewidth=0.7, linestyle=":", zorder=1)
    ax.scatter(size, y, s=22, color=GREY, edgecolor=INK, linewidth=0.6, zorder=3)
    for x, value in zip(size, y):
        above = value >= a + b * x
        side = -np.sign(b) if above else np.sign(b)
        ax.annotate(str(int(x)), (x, value), textcoords="offset points",
                    xytext=(4 * side, 3 if above else -3),
                    ha="left" if side > 0 else "right", va="bottom" if above else "top",
                    fontsize=TEXT, color=MUTED)
    ax.set_xlim(0.4, 6.6)
    ax.set_ylim(0.2, 7.0)
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(1, 7))
    ax.set_xlabel("Rank by size", fontsize=TEXT, color=INK)
    ax.set_ylabel(ylabel, fontsize=TEXT, color=INK)
    ax.set_title(title, loc="left", fontsize=TEXT, color=INK, pad=5)

ax = axes[2]
ax.axhline(0, color=RULE, linewidth=0.7, zorder=1)
ax.axvline(0, color=RULE, linewidth=0.7, zorder=1)
ax.scatter(cases_line[2], shortfall_line[2], s=22, color=GREY, edgecolor=INK, linewidth=0.6,
           zorder=3)
for x, (u, v) in enumerate(zip(cases_line[2], shortfall_line[2]), start=1):
    ax.annotate(str(x), (u, v), textcoords="offset points", xytext=(4, 3), fontsize=TEXT,
                color=MUTED)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel("Leftover of confirmed cases", fontsize=TEXT, color=INK)
ax.set_ylabel("Leftover of shortfall", fontsize=TEXT, color=INK)
ax.set_title("(c) The two leftovers", loc="left", fontsize=TEXT, color=INK, pad=5)
for ax in axes:
    ax.tick_params(labelsize=TEXT, colors=INK, length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
figure.savefig(OUT)
print("saved", OUT)
print(f"raw rank correlation {raw:+.3f}")
print(f"cases line {cases_line[0]:.2f} + {cases_line[1]:.3f} x size; leftovers "
      f"{np.round(cases_line[2], 2)}")
print(f"shortfall line {shortfall_line[0]:.2f} + {shortfall_line[1]:.3f} x size; leftovers "
      f"{np.round(shortfall_line[2], 2)}")
print(f"partial rank correlation {partial:+.3f}")
