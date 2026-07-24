# -*- coding: utf-8 -*-
"""Two-step dehomogenization (3-D recovery) flowchart for docs/architecture.md.

Same matplotlib-box idiom as ``scripts/build_iea22_flowchart.py``.  Writes
``docs/_img/dehom_flowchart.png`` relative to the repository root, so it is
runnable from anywhere:

    python scripts/build_dehom_flowchart.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "docs", "_img", "dehom_flowchart.png"))

BLUE, ORANGE, GREEN, PURPLE, TEAL = "#cfe3f3", "#fde0c0", "#cfead0", "#e3d4ef", "#bfe6e6"

fig, ax = plt.subplots(figsize=(15.0, 4.4))
ax.set_xlim(0, 17.8)
ax.set_ylim(0, 5.2)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=12.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.10",
                                fc=fc, ec="0.35", lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight,
            linespacing=1.5)


def proc(x, y, w, h, head, sub, fc):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.10",
                                fc=fc, ec="0.35", lw=1.3))
    ax.text(x, y + h * 0.28, head, ha="center", va="center", fontsize=13.5, weight="bold")
    ax.text(x, y - h * 0.17, sub, ha="center", va="center", fontsize=10.5, linespacing=1.55)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.5, color="0.35"))


# --- upper row: what the homogenization already produced -------------------
box(5.30, 4.15, 4.10, 1.10, "RM ring homogenization\n"
                            r"$C_6$,  warping $V_0$, $V_1$", PURPLE, 12.0)
box(12.75, 4.15, 4.10, 1.10, "MSG-RM $8\\times8$ wall law\n"
                             r"$[\mathbf{A},\mathbf{B};\,\mathbf{B},\mathbf{D}]$, $\mathbf{G}$"
                             "  + plate-SG warping", PURPLE, 12.0)

# --- main chain ------------------------------------------------------------
box(1.55, 1.60, 2.70, 1.35, "Beam resultants\n"
                            r"$F=[F_1,F_2,F_3,M_1,M_2,M_3]$", BLUE, 11.0)
proc(5.30, 1.60, 4.10, 1.70, "Step 1   beam $\\rightarrow$ shell",
     r"$\varepsilon = C_6^{-1}F$" "\n"
     "$V_0,V_1$ per laminate region\n(gradient-consistent)", ORANGE)
box(9.05, 1.60, 2.60, 1.35, "shell strains\n"
                            r"$s_6$, $s_2$ along $s$", GREEN, 12.0)
proc(12.75, 1.60, 4.10, 1.70, "Step 2   shell $\\rightarrow$ 3-D",
     r"$\Sigma(z)=C_{\rm layer}(z)\,[\,\mathbf{B}(z)V_0+G_e(z)\,]\,\varepsilon$" "\n"
     "same through-thickness SG as the ABD", ORANGE)
box(16.45, 1.60, 2.35, 1.60, "3-D $\\sigma$, $\\epsilon$\n"
                             "at $(y_2,y_3,z)$\n"
                             r"and $u=u_g+C(w{+}r)-r$", TEAL, 10.5)

arrow(2.95, 1.60, 3.20, 1.60)
arrow(7.40, 1.60, 7.70, 1.60)
arrow(10.40, 1.60, 10.65, 1.60)
arrow(14.85, 1.60, 15.22, 1.60)
arrow(5.30, 3.55, 5.30, 2.50)
arrow(12.75, 3.55, 12.75, 2.50)

# energy-consistency tie-back between the two steps
ax.annotate("", xy=(12.75, 0.52), xytext=(5.30, 0.52),
            arrowprops=dict(arrowstyle="<->", lw=1.2, color="0.55",
                            linestyle=(0, (4, 3))))
ax.text(9.03, 0.24, r"energy-consistent:  $\int \Gamma\!:\!\Sigma\,dz "
                    r"= \varepsilon^{\top}\mathbf{ABD}\,\varepsilon$",
        ha="center", va="center", fontsize=10.5, color="0.35")

fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
