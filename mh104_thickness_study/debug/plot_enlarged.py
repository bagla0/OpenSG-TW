"""Enlarged OML Timoshenko plots (f=0.1..0.6), with the non-diagonal terms split by block:
  * EB-block couplings (among EA,GJ,EI2,EI3 -- NO transverse shear)  -> 4-way (incl. FEniCS-shell)
  * transverse-shear couplings (involving GA2 or GA3)                -> 3-way (no FEniCS-shell)
Diagonal terms are kept (4-way default / 3-way with `nofe`).  Terms whose |solid| never exceeds
~5e6 (the "1e6 range", significantly smaller) are DROPPED.  Dashed line at f=0.3 = thin/thick
boundary (spar-cap h/H ~ 0.08, nearing the thin-shell limit h/H=0.1 at f~0.4).  abs + %error.

Run:  python plot_enlarged.py        -> 4-way (diagonal + EB couplings) -> mh104_thickness_study/figures
      python plot_enlarged.py nofe   -> 3-way (diagonal + shear couplings) -> oml_mh104_jax_vs_solid/figures
"""
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
FS = [10, 20, 30, 40, 60, 75]; fv = [x / 100 for x in FS]
KC, RC, FC = "tab:blue", "tab:red", "tab:green"
THIN = 0.3          # thin/thick boundary (geometry: spar h/H~0.08; h/H=0.1 at f~0.4)
THRESH = 1e7        # drop terms whose |solid| never exceeds this (i.e. stuck in the ~1e6 range)
EBset = {0, 3, 4, 5}


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


K = {fi: Ld(os.path.join(RES, "C6_shell_jax_OML_f%03d.txt" % fi)) for fi in FS}
R = {fi: Ld(os.path.join(RES, "C6_shell_rm_OML_f%03d.txt" % fi)) for fi in FS}
F = {fi: Ld(os.path.join(RES, "C6_fenics_shell_OML_f%03d.txt" % fi)) for fi in FS}
S = {fi: Ld(os.path.join(RES, "C6_solid_f%03d.txt" % fi)) for fi in FS}


def keep(i, j):
    return max((abs(S[fi][i, j]) for fi in FS if S[fi] is not None), default=0) >= THRESH


def gv(D, i, j):
    return [D[fi][i, j] if D[fi] is not None else np.nan for fi in FS]


def er(D, i, j):
    return [100 * (D[fi][i, j] - S[fi][i, j]) / abs(S[fi][i, j])
            if (D[fi] is not None and S[fi] is not None and abs(S[fi][i, j]) > 0) else np.nan for fi in FS]


od = [(i, j) for i in range(6) for j in range(i)]
diag = [(i, i) for i in range(6)]                 # all 6 diagonal -- never skip
coup = [(i, j) for (i, j) in od if keep(i, j)]    # couplings >= THRESH (drop the ~1e6 range)


def grid(terms, mode, fname, title):
    n = len(terms)
    ncol = min(4, n) if n else 1; nrow = int(np.ceil(n / ncol)) if n else 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.7 * nrow), squeeze=False)
    axf = axs.flat
    for k, (i, j) in enumerate(terms):
        ax = axf[k]
        if mode == "abs":
            ax.plot(fv, gv(K, i, j), "-o", color=KC, ms=5, lw=1.8, label="JAX-Kirchhoff")
            ax.plot(fv, gv(R, i, j), "-^", color=RC, ms=5, lw=1.8, label="JAX-RM")
            if INCLUDE_FE:
                ax.plot(fv, gv(F, i, j), "-d", color=FC, ms=5, lw=1.8, label="FEniCS-shell")
            ax.plot(fv, gv(S, i, j), "k-s", ms=6, lw=2.2, label="FEniCS-solid")
        else:
            ax.plot(fv, er(K, i, j), "-o", color=KC, ms=5, lw=1.8, label="JAX-Kirchhoff")
            ax.plot(fv, er(R, i, j), "-^", color=RC, ms=5, lw=1.8, label="JAX-RM")
            if INCLUDE_FE:
                ax.plot(fv, er(F, i, j), "-d", color=FC, ms=5, lw=1.8, label="FEniCS-shell")
            ax.axhline(0, color="k", lw=0.8); ax.axhspan(-5, 5, color="0.85", alpha=0.6)
            ax.set_ylabel("% error", fontsize=9)
        if i == j:
            ax.set_title("%s  (C%d%d)" % (lab[i], i + 1, i + 1), fontsize=12, fontweight="bold")
        else:
            a, b = min(i, j), max(i, j)
            ax.set_title("C%d%d:  %s-%s" % (a + 1, b + 1, lab[a], lab[b]), fontsize=12)
        ax.axvline(THIN, color="0.45", ls="--", lw=1.3, zorder=0)
        ax.set_xlabel("thickness factor f", fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
    for k in range(n, nrow * ncol):
        axf[k].axis("off")
    axf[0].legend(fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.01, "dashed line = f=0.3 thin/thick boundary (spar h/H~0.08);  thinner walls left, thicker right",
             ha="center", fontsize=10, style="italic", color="0.3")
    fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")


fe = " / FEniCS-shell" if INCLUDE_FE else ""
tag = "JAX-Kirchhoff / JAX-RM%s / FEniCS-solid" % fe
grid(diag, "abs", "oml_abs_diagonal.png", "mh104 OML diagonal (4 way): " + tag)
grid(coup, "abs", "oml_abs_coupling.png", "mh104 OML couplings (|solid|>=1e7): " + tag)
grid(diag, "pct", "oml_pcterr_diagonal.png", "mh104 OML diagonal %% error vs solid (grey +-5%)")
grid(coup, "pct", "oml_pcterr_coupling.png", "mh104 OML couplings %% error vs solid")
dropped = [(i, j) for (i, j) in od if not keep(i, j)]
print("%s: diagonal=ALL 6  couplings>=1e7=%s" % ("4-way" if INCLUDE_FE else "3-way", ["C%d%d" % (min(i, j) + 1, max(i, j) + 1) for i, j in coup]))
print("dropped couplings (|solid|<%.0e): %s" % (THRESH, ["C%d%d" % (min(i, j) + 1, max(i, j) + 1) for i, j in dropped]))
