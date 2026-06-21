"""Enlarged plots of ALL nonzero Timoshenko 6x6 terms vs thickness factor f, OML reference ONLY.
Two flavors per group: absolute (shell OML vs solid) and % error (shell OML vs solid).
Groups: diagonal (6) and off-diagonal coupling (15, split in two).  Off-diagonal terms whose
|solid| stays < 1% of EA are tagged '(small)' (their % error is noisy)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mh104_thickness_study"
RES = os.path.join(STUDY, "results"); FIG = os.path.join(STUDY, "figures")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 40, 60, 80, 100]; fv = [x / 100 for x in FS]
OMLc = "tab:blue"


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


shell = {fi: Ld(os.path.join(RES, "C6_shell_jax_OML_f%03d.txt" % fi)) for fi in FS}
solid = {fi: Ld(os.path.join(RES, "C6_solid_f%03d.txt" % fi)) for fi in FS}
diag = [(i, i) for i in range(6)]
od = [(i, j) for i in range(6) for j in range(i)]


def is_small(i, j):
    return all((solid[fi] is None) or (abs(solid[fi][i, j]) < 0.01 * solid[fi][0, 0]) for fi in FS)


def grid(terms, nrow, ncol, mode, fname, title, figsize):
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axf = np.atleast_1d(axs).flat
    for k, (i, j) in enumerate(terms):
        ax = axf[k]; small = (mode == "pct") and is_small(i, j)
        if mode == "abs":
            y = [shell[fi][i, j] if shell[fi] is not None else np.nan for fi in FS]
            ax.plot(fv, y, "-o", color=OMLc, ms=5, lw=1.8, label="shell OML")
            ys = [solid[fi][i, j] if solid[fi] is not None else np.nan for fi in FS]
            ax.plot(fv, ys, "k--s", ms=6, lw=2, label="solid")
        else:
            y = [100 * (shell[fi][i, j] - solid[fi][i, j]) / abs(solid[fi][i, j])
                 if (shell[fi] is not None and solid[fi] is not None and abs(solid[fi][i, j]) > 0) else np.nan for fi in FS]
            ax.plot(fv, y, "-o", color=OMLc, ms=5, lw=1.8)
            ax.axhline(0, color="k", lw=0.8); ax.axhspan(-5, 5, color="0.85", alpha=0.6); ax.set_ylim(-80, 80)
            ax.set_ylabel("% error", fontsize=9)
            if small:
                ax.set_facecolor("0.95")
        ttl = (lab[i] if i == j else "%s-%s" % (lab[i], lab[j])) + (" (small)" if small else "")
        ax.set_title(ttl, fontsize=12, fontweight="bold" if i == j else "normal", color="0.5" if small else "k")
        ax.set_xlabel("thickness factor f", fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
    for k in range(len(terms), nrow * ncol):
        axf[k].axis("off")
    if mode == "abs":
        axf[0].legend(fontsize=11)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")


# ---- absolute (shell OML vs solid) ----
grid(diag, 2, 3, "abs", "oml_abs_diagonal.png", "mh104 Timoshenko DIAGONAL vs f  (shell OML vs solid)", (17, 9))
grid(od[:8], 2, 4, "abs", "oml_abs_coupling_1.png", "mh104 OFF-DIAGONAL coupling vs f  (1/2, shell OML vs solid)", (20, 9))
grid(od[8:], 2, 4, "abs", "oml_abs_coupling_2.png", "mh104 OFF-DIAGONAL coupling vs f  (2/2, shell OML vs solid)", (20, 9))
# ---- % error (shell OML vs solid) ----
grid(diag, 2, 3, "pct", "oml_pcterr_diagonal.png", "mh104 % ERROR DIAGONAL  (shell OML vs solid; grey +-5%)", (17, 9))
grid(od[:8], 2, 4, "pct", "oml_pcterr_coupling_1.png", "mh104 % ERROR OFF-DIAGONAL  (1/2, OML; 'small'=noisy)", (20, 9))
grid(od[8:], 2, 4, "pct", "oml_pcterr_coupling_2.png", "mh104 % ERROR OFF-DIAGONAL  (2/2, OML; 'small'=noisy)", (20, 9))
print("wrote oml_abs_{diagonal,coupling_1,coupling_2}.png and oml_pcterr_{diagonal,coupling_1,coupling_2}.png")
