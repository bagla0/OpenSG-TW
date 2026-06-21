"""Enlarged OML plots of ALL 21 nonzero Timoshenko terms vs thickness factor f (0.1..0.6).
Default: 4 models (JAX-Kirchhoff, JAX-RM, FEniCS-shell, FEniCS-solid) -> mh104_thickness_study/figures.
`nofe` arg: drop FEniCS-shell (it collapses thin-wall flapwise terms) -> oml_mh104_jax_vs_solid/figures.
ABS figures = absolute stiffness; %ERROR figures = (shell - solid)/solid.  Off-diagonal panels are
labelled with the 6x6 bracket index C_ij and the stiffness pair.  Dashed line at f=0.2 = nominal
(validated) mh104 thickness, thinner to the left / thicker to the right."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mh104_thickness_study"
RES = os.path.join(STUDY, "results"); FIG = os.path.join(STUDY, "figures")
NOFE = "nofe" in sys.argv
INCLUDE_FE = not NOFE
if NOFE:
    FIG = os.path.join(os.path.dirname(STUDY), "oml_mh104_jax_vs_solid", "figures")
    os.makedirs(FIG, exist_ok=True)
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 40, 60]; fv = [x / 100 for x in FS]
KC, RC, FC = "tab:blue", "tab:red", "tab:green"


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


K = {fi: Ld(os.path.join(RES, "C6_shell_jax_OML_f%03d.txt" % fi)) for fi in FS}
R = {fi: Ld(os.path.join(RES, "C6_shell_rm_OML_f%03d.txt" % fi)) for fi in FS}
F = {fi: Ld(os.path.join(RES, "C6_fenics_shell_OML_f%03d.txt" % fi)) for fi in FS}
S = {fi: Ld(os.path.join(RES, "C6_solid_f%03d.txt" % fi)) for fi in FS}
diag = [(i, i) for i in range(6)]
od = [(i, j) for i in range(6) for j in range(i)]


def is_small(i, j):
    return all((S[fi] is None) or (abs(S[fi][i, j]) < 0.01 * S[fi][0, 0]) for fi in FS)


def gv(D, i, j):
    return [D[fi][i, j] if D[fi] is not None else np.nan for fi in FS]


def er(D, i, j):
    return [100 * (D[fi][i, j] - S[fi][i, j]) / abs(S[fi][i, j])
            if (D[fi] is not None and S[fi] is not None and abs(S[fi][i, j]) > 0) else np.nan for fi in FS]


def grid(terms, nrow, ncol, mode, fname, title, figsize):
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axf = np.atleast_1d(axs).flat
    for k, (i, j) in enumerate(terms):
        ax = axf[k]; small = is_small(i, j)
        if mode == "abs":
            ax.plot(fv, gv(K, i, j), "-o", color=KC, ms=5, lw=1.8, label="JAX-Kirchhoff")
            ax.plot(fv, gv(R, i, j), "-^", color=RC, ms=5, lw=1.8, label="JAX-RM")
            if INCLUDE_FE:
                ax.plot(fv, gv(F, i, j), "-d", color=FC, ms=5, lw=1.8, label="FEniCS-shell")
            ax.plot(fv, gv(S, i, j), "k--s", ms=6, lw=2, label="FEniCS-solid")
        else:
            ax.plot(fv, er(K, i, j), "-o", color=KC, ms=5, lw=1.8, label="JAX-Kirchhoff")
            ax.plot(fv, er(R, i, j), "-^", color=RC, ms=5, lw=1.8, label="JAX-RM")
            if INCLUDE_FE:
                ax.plot(fv, er(F, i, j), "-d", color=FC, ms=5, lw=1.8, label="FEniCS-shell")
            ax.axhline(0, color="k", lw=0.8); ax.axhspan(-5, 5, color="0.85", alpha=0.6)
            ax.set_ylabel("% error", fontsize=9)
            if small:
                ax.set_ylim(-120, 120); ax.set_facecolor("0.95")
        if i == j:
            base = "%s  (C%d%d)" % (lab[i], i + 1, i + 1)
        else:
            a, b = min(i, j), max(i, j)
            base = "C%d%d:  %s-%s" % (a + 1, b + 1, lab[a], lab[b])
        ttl = base + (" (small)" if (mode == "pct" and small) else "")
        ax.set_title(ttl, fontsize=12, fontweight="bold" if i == j else "normal", color="0.5" if (mode == "pct" and small) else "k")
        ax.axvline(0.2, color="0.45", ls="--", lw=1.3, zorder=0)   # f=0.2 = nominal mh104 (validated) thickness
        ax.set_xlabel("thickness factor f", fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
    for k in range(len(terms), nrow * ncol):
        axf[k].axis("off")
    axf[0].legend(fontsize=10)
    fig.suptitle(title, fontsize=15)
    fig.text(0.5, 0.006, "dashed line = f=0.2 (nominal mh104 thickness);  thinner walls to the left, thicker to the right",
             ha="center", fontsize=10, style="italic", color="0.3")
    fig.tight_layout(rect=[0, 0.02, 1, 1]); fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")


fe = " / FEniCS-shell" if INCLUDE_FE else ""
A = "mh104 OML Timoshenko: JAX-Kirchhoff / JAX-RM%s / FEniCS-solid" % fe
grid(diag, 2, 3, "abs", "oml_abs_diagonal.png", A + " - DIAGONAL", (17, 9))
grid(od[:8], 2, 4, "abs", "oml_abs_coupling_1.png", A + " - coupling 1/2", (20, 9))
grid(od[8:], 2, 4, "abs", "oml_abs_coupling_2.png", A + " - coupling 2/2", (20, 9))
P = "mh104 OML %% ERROR vs solid: JAX-Kirchhoff / JAX-RM%s" % fe
grid(diag, 2, 3, "pct", "oml_pcterr_diagonal.png", P + " - DIAGONAL (grey +-5%)", (17, 9))
grid(od[:8], 2, 4, "pct", "oml_pcterr_coupling_1.png", P + " - coupling 1/2", (20, 9))
grid(od[8:], 2, 4, "pct", "oml_pcterr_coupling_2.png", P + " - coupling 2/2", (20, 9))
print("wrote %s plots to %s" % ("3-way (no FEniCS-shell)" if NOFE else "4-way", FIG))
