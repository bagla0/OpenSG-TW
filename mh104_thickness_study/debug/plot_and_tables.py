"""Plot every nonzero Timoshenko term vs thickness factor f for OML/center/IML (JAX-Kirchhoff shell
vs FEniCS solid), and write the full comparison tables to a .txt.  Reads results/C6_*.txt."""
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
have_sol = [fi for fi in FS if solid[fi] is not None]
print("solid refs available for f =", [x / 100 for x in have_sol])

# ---- Figure 1: absolute diagonal stiffness vs f ----
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for ti in range(6):
    ax = axs.flat[ti]
    for r in REFS:
        y = [shell[(fi, r)][ti, ti] if shell[(fi, r)] is not None else np.nan for fi in FS]
        ax.plot(fv, y, "-o", color=col[r], ms=4, label="shell %s" % r)
    ys = [solid[fi][ti, ti] if solid[fi] is not None else np.nan for fi in FS]
    ax.plot(fv, ys, "k--s", lw=2, ms=6, label="solid (VABS-val.)")
    ax.set_title(lab[ti], fontsize=12); ax.set_xlabel("thickness factor f"); ax.grid(alpha=0.3)
    if ti == 0:
        ax.legend(fontsize=8)
fig.suptitle("mh104 Timoshenko diagonal vs wall-thickness factor  (JAX-Kirchhoff shell vs FEniCS solid)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "timo_diag_abs.png"), dpi=140, bbox_inches="tight")
print("wrote timo_diag_abs.png")

# ---- Figure 2: % diff (shell - solid)/solid vs f ----
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for ti in range(6):
    ax = axs.flat[ti]
    for r in REFS:
        y = [100 * (shell[(fi, r)][ti, ti] - solid[fi][ti, ti]) / abs(solid[fi][ti, ti])
             if (solid[fi] is not None and shell[(fi, r)] is not None) else np.nan for fi in FS]
        ax.plot(fv, y, "-o", color=col[r], ms=4, label=r)
    ax.axhline(0, color="k", lw=0.8); ax.axhspan(-5, 5, color="0.85", alpha=0.5)
    ax.set_title(lab[ti], fontsize=12); ax.set_xlabel("thickness factor f"); ax.set_ylabel("% diff vs solid")
    ax.grid(alpha=0.3); ax.set_ylim(-30, 90)   # cap so IML thick-wall breakdown spikes (GA3/EI2) don't compress the rest
    if ti == 0:
        ax.legend(fontsize=8, title="shell ref")
fig.suptitle("mh104 shell vs solid: Timoshenko diagonal %diff (grey band = +-5%)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "timo_diag_pctdiff.png"), dpi=140, bbox_inches="tight")
print("wrote timo_diag_pctdiff.png")

# ---- comparison tables .txt ----
with open(os.path.join(RES, "timo_comparison_tables.txt"), "w") as fh:
    fh.write("mh104 thickness sweep: Timoshenko 6x6, JAX-Kirchhoff shell (CCW, k22=0) vs FEniCS solid.\n")
    fh.write("Order [EA, GA2, GA3, GJ, EI2, EI3].  Reference surfaces: OML(frac0)/center(0.5)/IML(1.0).\n\n")
    for fi in FS:
        f = fi / 100
        fh.write("===== f = %.2f =====\n" % f)
        if solid[fi] is not None:
            fh.write("  solid diag:  " + "  ".join("%s=%.4e" % (lab[i], solid[fi][i, i]) for i in range(6)) + "\n")
        else:
            fh.write("  solid diag:  (not computed)\n")
        for r in REFS:
            C = shell[(fi, r)]
            if C is None:
                continue
            fh.write("  shell %-6s " % r + "  ".join("%s=%.4e" % (lab[i], C[i, i]) for i in range(6)) + "\n")
            if solid[fi] is not None:
                fh.write("    %%diff      " % () + "  ".join("%s=%+6.1f%%" % (lab[i], 100 * (C[i, i] - solid[fi][i, i]) / abs(solid[fi][i, i])) for i in range(6)) + "\n")
        fh.write("\n")
print("wrote timo_comparison_tables.txt")
