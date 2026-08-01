"""fig_conventions.py -- the manual's conventions sketch: stacking direction,
ply numbering (bottom first), fiber-angle rotation about x3, and the
reference-surface choices (fraction = 0 / 0.5 / 1).  Writes conventions.png
next to this script (no title, per the figure conventions).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                              gridspec_kw={"width_ratios": [1.15, 1.0]})

# ---- left panel: the stack ----
plies = [("ply 1  (BOTTOM)", 0.00, 0.30, "#c6dbef", r"$\theta_1$"),
         ("ply 2", 0.30, 0.65, "#9ecae1", r"$\theta_2$"),
         ("ply 3  (TOP)", 0.65, 1.00, "#6baed6", r"$\theta_3$")]
for name, z0, z1, col, th in plies:
    ax.fill_between([0, 1], z0, z1, color=col, edgecolor="k", lw=0.8)
    ax.text(0.03, 0.5 * (z0 + z1), name, fontsize=10, va="center")
    ax.text(0.78, 0.5 * (z0 + z1), th, fontsize=11, va="center")
for frac, lbl in ((0.0, "fraction = 0  (bottom / OML face)"),
                  (0.5, "fraction = 0.5  (mid-surface)"),
                  (1.0, "fraction = 1  (top / IML face)")):
    ax.axhline(frac, color="crimson", lw=1.2, ls="--")
    ax.text(1.02, frac, lbl, fontsize=9, va="center", color="crimson")
ax.annotate("", xy=(-0.06, 1.05), xytext=(-0.06, -0.05),
            arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(-0.11, 0.5, r"$x_3$", fontsize=13, va="center")
ax.annotate("", xy=(1.0, -0.12), xytext=(0.0, -0.12),
            arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(0.5, -0.20, r"$x_1$", fontsize=13, ha="center")
ax.set_xlim(-0.15, 1.75)
ax.set_ylim(-0.28, 1.12)
ax.axis("off")

# ---- right panel: the fiber angle ----
ax2.annotate("", xy=(1.05, 0), xytext=(-1.05, 0),
             arrowprops=dict(arrowstyle="->", lw=1.4))
ax2.annotate("", xy=(0, 1.05), xytext=(0, -1.05),
             arrowprops=dict(arrowstyle="->", lw=1.4))
ax2.text(1.08, 0.02, r"$x_1$", fontsize=13)
ax2.text(0.03, 1.08, r"$x_2$", fontsize=13)
th = np.deg2rad(30.0)
ax2.plot([-np.cos(th), np.cos(th)], [-np.sin(th), np.sin(th)],
         color="#d62728", lw=2.4)
ax2.text(np.cos(th) + 0.04, np.sin(th) + 0.02, "fiber (material 1-axis)",
         fontsize=10, color="#d62728")
arc = np.linspace(0, th, 40)
ax2.plot(0.55 * np.cos(arc), 0.55 * np.sin(arc), color="k", lw=1.0)
ax2.text(0.62 * np.cos(th / 2), 0.62 * np.sin(th / 2) - 0.02,
         r"$\theta$ (+ about $x_3$)", fontsize=11)
ax2.set_xlim(-1.25, 1.6)
ax2.set_ylim(-1.2, 1.25)
ax2.set_aspect("equal")
ax2.axis("off")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "conventions.png"), dpi=160,
            bbox_inches="tight")
print("wrote conventions.png")
