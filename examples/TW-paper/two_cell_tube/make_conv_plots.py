"""Per-term percentage-error bar charts (vs FEniCS-2D-solid) for the two-cell
[-45] curved tube, thin vs thick, one figure per shell model (RM, KL).  This is
the thin/thick convergence view that complements the aniso table.
  -> figures/conv_2cell_aniso_RM.png, conv_2cell_aniso_KL.png"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)
from gradient_kirchhoff import gradient_junction_kirchhoff
from strip_RM import rm_timoshenko_6x6

DATA = os.path.join(CC, "multicell_tube", "data")
FIG = os.path.join(CC, "multicell_tube", "figures")
TERMS = [(r"$EA$", 0, 0), (r"$GA_2$", 1, 1), (r"$GA_3$", 2, 2), (r"$GJ$", 3, 3),
         (r"$EI_2$", 4, 4), (r"$EI_3$", 5, 5), (r"$C_{14}$", 0, 3)]
CTHIN, CTHICK = "#4C72B0", "#C44E52"


def L(n):
    M = np.loadtxt(os.path.join(DATA, n))
    return 0.5 * (M + M.T)


def errs(solidf, shellf, t):
    S = L(solidf)
    mesh = os.path.join(DATA, shellf)
    KF, _, _ = gradient_junction_kirchhoff(mesh, frac=0.0, dshift=t / 2.0)
    KF = 0.5 * (np.asarray(KF) + np.asarray(KF).T)
    RM = np.asarray(rm_timoshenko_6x6(mesh, 0.0, dshift=t / 2.0, curved=True))
    RM = 0.5 * (RM + RM.T)
    return S, KF, RM


def pe(M, S, i, j):
    return 100.0 * (M[i, j] - S[i, j]) / S[i, j]


thin = errs("C6_solid_tube2cell_aniso_thin.txt", "tube2cell_aniso_thin.yaml", 0.004)
thick = errs("C6_solid_tube2cell_aniso_thick.txt", "tube2cell_aniso_thick.yaml", 0.016)

for model, k in [("RM", 2), ("KL", 1)]:
    et = [pe(thin[k], thin[0], i, j) for (_, i, j) in TERMS]
    ek = [pe(thick[k], thick[0], i, j) for (_, i, j) in TERMS]
    x = np.arange(len(TERMS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    b1 = ax.bar(x - w / 2, et, w, label=r"thin ($R/h=12.5$)", color=CTHIN)
    b2 = ax.bar(x + w / 2, ek, w, label=r"thick ($R/h=3.1$)", color=CTHICK)
    ax.axhline(0, color="k", lw=0.8)
    ax.bar_label(b1, fmt="%+.1f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%+.1f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in TERMS], fontsize=12)
    ax.set_ylabel("% error vs. 2-D solid", fontsize=12)
    ax.set_title(model, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.margins(y=0.16)
    fig.tight_layout()
    out = os.path.join(FIG, "conv_2cell_aniso_%s.png" % model)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))
