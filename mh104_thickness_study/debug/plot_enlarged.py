"""Enlarged OML plots of ALL nonzero Timoshenko 6x6 terms vs thickness factor f, comparing the two JAX
shell models -- Kirchhoff (C1-Hermite) and Reissner-Mindlin -- against the FEniCS solid.
Absolute figures (Kirchhoff, RM, solid) and % error figures (Kirchhoff, RM vs solid).
Groups: diagonal (6) and off-diagonal coupling (15, split in two)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mh104_thickness_study"
RES = os.path.join(STUDY, "results"); FIG = os.path.join(STUDY, "figures")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 40, 60, 80, 100]; fv = [x / 100 for x in FS]
KC, RC = "tab:blue", "tab:red"


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


K = {fi: Ld(os.path.join(RES, "C6_shell_jax_OML_f%03d.txt" % fi)) for fi in FS}   # Kirchhoff
R = {fi: Ld(os.path.join(RES, "C6_shell_rm_OML_f%03d.txt" % fi)) for fi in FS}    # Reissner-Mindlin
S = {fi: Ld(os.path.join(RES, "C6_solid_f%03d.txt" % fi)) for fi in FS}           # solid
diag = [(i, i) for i in range(6)]
od = [(i, j) for i in range(6) for j in range(i)]


def is_small(i, j):
    return all((S[fi] is None) or (abs(S[fi][i, j]) < 0.01 * S[fi][0, 0]) for fi in FS)


def gv(D, i, j):
    return [D[fi][i, j] if D[fi] is not None else np.nan for fi in FS]


def err(D, i, j):
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
            ax.plot(fv, gv(S, i, j), "k--s", ms=6, lw=2, label="FEniCS solid")
        else:
            ax.plot(fv, err(K, i, j), "-o", color=KC, ms=5, lw=1.8, label="Kirchhoff")
            ax.plot(fv, err(R, i, j), "-^", color=RC, ms=5, lw=1.8, label="RM")
            ax.axhline(0, color="k", lw=0.8); ax.axhspan(-5, 5, color="0.85", alpha=0.6)
            ax.set_ylabel("% error", fontsize=9)
            if small:
                ax.set_ylim(-120, 120); ax.set_facecolor("0.95")
        ttl = (lab[i] if i == j else "%s-%s" % (lab[i], lab[j])) + (" (small)" if (mode == "pct" and small) else "")
        ax.set_title(ttl, fontsize=12, fontweight="bold" if i == j else "normal", color="0.5" if (mode == "pct" and small) else "k")
        ax.set_xlabel("thickness factor f", fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
    for k in range(len(terms), nrow * ncol):
        axf[k].axis("off")
    axf[0].legend(fontsize=11)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")


T = "mh104 OML: JAX-Kirchhoff vs JAX-RM vs FEniCS solid"
grid(diag, 2, 3, "abs", "oml_abs_diagonal.png", T + " - DIAGONAL", (17, 9))
grid(od[:8], 2, 4, "abs", "oml_abs_coupling_1.png", T + " - coupling 1/2", (20, 9))
grid(od[8:], 2, 4, "abs", "oml_abs_coupling_2.png", T + " - coupling 2/2", (20, 9))
T2 = "mh104 OML % ERROR vs solid: Kirchhoff vs RM"
grid(diag, 2, 3, "pct", "oml_pcterr_diagonal.png", T2 + " - DIAGONAL (grey +-5%)", (17, 9))
grid(od[:8], 2, 4, "pct", "oml_pcterr_coupling_1.png", T2 + " - coupling 1/2", (20, 9))
grid(od[8:], 2, 4, "pct", "oml_pcterr_coupling_2.png", T2 + " - coupling 2/2", (20, 9))
print("wrote 3-way oml_abs_* and oml_pcterr_* (Kirchhoff/RM/solid)")
