"""One PNG per individual Timoshenko term (all 21: 6 diagonal + 15 coupling), 4 curves
(JAX-Kirchhoff/RM/FEniCS-shell + FEniCS-solid benchmark star), for dropping into Overleaf.
Saved to oml_mh104_4way/individual/.  Filenames are the bracket index, e.g. C11_EA.png, C14_EA-GJ.png."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
RES = os.path.join(CC, "mh104_thickness_study", "results")
OUT = os.path.join(CC, "oml_mh104_4way", "individual"); os.makedirs(OUT, exist_ok=True)
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 30, 40, 60, 75]; fv = [x / 100 for x in FS]
KC, RC, FC = "tab:blue", "tab:red", "tab:green"


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


K = {fi: Ld(os.path.join(RES, "C6_shell_jax_OML_f%03d.txt" % fi)) for fi in FS}
R = {fi: Ld(os.path.join(RES, "C6_shell_rm_OML_f%03d.txt" % fi)) for fi in FS}
F = {fi: Ld(os.path.join(RES, "C6_fenics_shell_OML_f%03d.txt" % fi)) for fi in FS}
S = {fi: Ld(os.path.join(RES, "C6_solid_f%03d.txt" % fi)) for fi in FS}


def gv(D, i, j):
    return [D[fi][i, j] if D[fi] is not None else np.nan for fi in FS]


n = 0
for i in range(6):
    for j in range(i + 1):
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        ax.plot(fv, gv(K, i, j), "-o", color=KC, ms=6, lw=1.6, mfc="none", mew=1.6, label="JAX-Kirchhoff")
        ax.plot(fv, gv(R, i, j), "-^", color=RC, ms=6, lw=1.6, mfc="none", mew=1.6, label="JAX-RM")
        ax.plot(fv, gv(F, i, j), "-d", color=FC, ms=6, lw=1.6, mfc="none", mew=1.6, label="FEniCS-shell")
        ax.plot(fv, gv(S, i, j), "-", color="k", marker="*", ms=15, lw=2.6, zorder=6, label="FEniCS-solid (benchmark)")
        if i == j:
            name = "C%d%d_%s" % (i + 1, i + 1, lab[i]); title = "%s  (C%d%d)" % (lab[i], i + 1, i + 1)
        else:
            name = "C%d%d_%s-%s" % (j + 1, i + 1, lab[j], lab[i]); title = "C%d%d:  %s-%s" % (j + 1, i + 1, lab[j], lab[i])
        ax.axvline(0.3, color="0.45", ls="--", lw=1.3, zorder=0)
        ax.set_title(title, fontsize=13); ax.set_xlabel("thickness factor f", fontsize=11)
        ax.set_ylabel("stiffness", fontsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig); n += 1
print("wrote %d individual term plots to %s" % (n, OUT))
