"""A six-bank example of the out-of-fold prediction, for the appendix to Chapter 5.

Constructed values, not data: six banks with a property value of 1 to 6 (its
logarithm, say) and shortfalls of 0.5, 0.6, 0.2, 0.4, 0.3 and 0.1, divided into
three groups, {1, 4}, {2, 5} and {3, 6}. Each group is predicted by an
unweighted least-squares line fitted on the other four banks. Panel (a) draws
the three lines and the out-of-fold predictions on them; panel (b) plots the
out-of-fold predictions against the shortfalls. The lines, the predictions and
both rank correlations are printed so that the appendix text can be checked
against them.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "app_oof.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})

INK, MUTED, RULE = "#1c1c1c", "#6b6b6b", "#d7d3cb"
COLOURS = ("#2a5d9f", "#c2352c", "#4f8a3a")
STYLES = ("-", "--", ":")

value = np.arange(1, 7, dtype=float)
shortfall = np.array([0.5, 0.6, 0.2, 0.4, 0.3, 0.1])
groups = ((0, 3), (1, 4), (2, 5))

lines, prediction = [], np.empty(6)
for held in groups:
    fit = [i for i in range(6) if i not in held]
    slope, intercept = np.polyfit(value[fit], shortfall[fit], 1)
    lines.append((intercept, slope))
    prediction[list(held)] = intercept + slope * value[list(held)]
slope_all, intercept_all = np.polyfit(value, shortfall, 1)
rho_out = spearmanr(prediction, shortfall).statistic
rho_in = spearmanr(intercept_all + slope_all * value, shortfall).statistic

figure, axes = plt.subplots(1, 2, figsize=(6.3, 3.4),
                            gridspec_kw=dict(left=0.095, right=0.985, bottom=0.30, top=0.91,
                                             wspace=0.34))
ax = axes[0]
for (intercept, slope), held, colour, style in zip(lines, groups, COLOURS, STYLES):
    ax.plot([0.6, 6.4], [intercept + slope * 0.6, intercept + slope * 6.4], color=colour,
            lw=1.0, ls=style, zorder=2,
            label=f"Fitted without\nbanks {held[0] + 1} and {held[1] + 1}")
    ax.scatter(value[list(held)], prediction[list(held)], s=26, marker="s", facecolor="white",
               edgecolor=colour, lw=0.9, zorder=4)
ax.scatter(value, shortfall, s=22, color="#c3bfb6", edgecolor=INK, lw=0.6, zorder=3)
label_offset = {1: (5, -3), 5: (4, -10), 6: (4, -10)}
for x, y in zip(value, shortfall):
    ax.annotate(str(int(x)), (x, y), textcoords="offset points",
                xytext=label_offset.get(int(x), (4, 3)), fontsize=10, color=MUTED)
ax.set_xlabel("Property value", fontsize=10.5, color=INK)
ax.set_ylabel("Shortfall", fontsize=10.5, color=INK)
ax.set_title("(a) Three fits, each predicting two banks", fontsize=11, color=INK, loc="left")
figure.legend(*ax.get_legend_handles_labels(), fontsize=10, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 0.0), ncol=3, handlelength=2.2, columnspacing=2.0)
ax.set_ylim(0.0, 0.8)
ax.set_xticks(value)
ax.set_yticks(np.arange(0.0, 0.85, 0.1))

ax = axes[1]
for held, colour in zip(groups, COLOURS):
    ax.scatter(prediction[list(held)], shortfall[list(held)], s=26, marker="s",
               facecolor="white", edgecolor=colour, lw=0.9, zorder=3)
for i in range(6):
    ax.annotate(str(i + 1), (prediction[i], shortfall[i]), textcoords="offset points",
                xytext=(4, 3), fontsize=10, color=MUTED)
ax.text(0.04, 0.95, f"ρ = {rho_out:.3f}", transform=ax.transAxes, fontsize=10,
        color=MUTED, va="top")
ax.set_xlabel("Out-of-fold prediction", fontsize=10.5, color=INK)
ax.set_ylabel("Shortfall", fontsize=10.5, color=INK)
ax.set_title("(b) Predictions and shortfalls", fontsize=11, color=INK, loc="left")
ax.set_xlim(0.15, 0.62)
ax.set_ylim(0.0, 0.7)

for ax in axes:
    ax.tick_params(labelsize=10, color=MUTED, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)

figure.savefig(OUT)
print("saved", OUT)
for (intercept, slope), held in zip(lines, groups):
    print(f"without {held[0] + 1},{held[1] + 1}: {intercept:.4f} {slope:+.4f} x")
print("predictions", np.round(prediction, 4))
print(f"all six: {intercept_all:.4f} {slope_all:+.4f} x; in-sample rho {rho_in:.4f}; "
      f"out-of-fold rho {rho_out:.4f}")
