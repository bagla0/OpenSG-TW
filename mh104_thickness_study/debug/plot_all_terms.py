"""Plot ALL nonzero Timoshenko 6x6 terms vs thickness factor f, laid out as the 6x6 matrix
(lower triangle = 21 unique terms).  Each panel: JAX-Kirchhoff shell at OML/center/IML (lines) vs
FEniCS solid (black markers).  Off-diagonals near zero are still shown."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mh104_thickness_study"
RES = os.path.join(STUDY, "results")
FIG = os.path.join(STUDY, "figures")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 40, 60, 80, 100]
fv = [x / 100 for x in FS]
REFS = ["OML", "center", "IML"]
col = {"OML": "tab:blue", "center": "tab:green", "IML": "tab:orange"}


def L(p):
    return np.loadtxt(p) if os.path.exists(p) else None


shell = {(fi, r): L(os.path.join(RES, "C6_shell_jax_%s_f%03d.txt" % (r, fi))) for fi in FS for r in REFS}
solid = {fi: L(os.path.join(RES, "C6_solid_f%03d.txt" % fi)) for fi in FS}

fig, axs = plt.subplots(6, 6, figsize=(20, 18))
for i in range(6):
    for j in range(6):
        ax = axs[i, j]
        if j > i:
            ax.axis("off"); continue
        for r in REFS:
            y = [shell[(fi, r)][i, j] if shell[(fi, r)] is not None else np.nan for fi in FS]
            ax.plot(fv, y, "-o", color=col[r], ms=3, lw=1.2, label=r)
        ys = [solid[fi][i, j] if solid[fi] is not None else np.nan for fi in FS]
        ax.plot(fv, ys, "k--s", ms=4, lw=1.5, label="solid")
        ax.set_title("%s-%s" % (lab[i], lab[j]) if i != j else lab[i],
                     fontsize=9, fontweight="bold" if i == j else "normal")
        ax.tick_params(labelsize=6); ax.grid(alpha=0.25)
        if i == 5:
            ax.set_xlabel("f", fontsize=8)
axs[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("mh104: ALL Timoshenko 6x6 terms vs wall-thickness factor f  "
             "(JAX-Kirchhoff shell OML/center/IML vs FEniCS solid)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(FIG, "timo_all_terms.png"), dpi=130, bbox_inches="tight")
print("wrote figures/timo_all_terms.png")
