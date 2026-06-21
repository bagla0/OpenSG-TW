"""Enlarged, readable plots of ALL nonzero Timoshenko terms vs thickness factor f, OML + center only
(IML set aside for debugging).  Three abs-value figures (diagonal, coupling-1, coupling-2) plus a
separate %error figure for the diagonal."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mh104_thickness_study"
RES = os.path.join(STUDY, "results"); FIG = os.path.join(STUDY, "figures")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 40, 60, 80, 100]; fv = [x / 100 for x in FS]
REFS = ["OML", "center"]; col = {"OML": "tab:blue", "center": "tab:green"}


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


shell = {(fi, r): Ld(os.path.join(RES, "C6_shell_jax_%s_f%03d.txt" % (r, fi))) for fi in FS for r in REFS}
solid = {fi: Ld(os.path.join(RES, "C6_solid_f%03d.txt" % fi)) for fi in FS}


def panel(ax, i, j):
    for r in REFS:
        y = [shell[(fi, r)][i, j] if shell[(fi, r)] is not None else np.nan for fi in FS]
        ax.plot(fv, y, "-o", color=col[r], ms=5, lw=1.8, label="shell " + r)
    ys = [solid[fi][i, j] if solid[fi] is not None else np.nan for fi in FS]
    ax.plot(fv, ys, "k--s", ms=6, lw=2, label="solid")
    ax.set_title(lab[i] if i == j else "%s-%s" % (lab[i], lab[j]), fontsize=13,
                 fontweight="bold" if i == j else "normal")
    ax.set_xlabel("thickness factor f", fontsize=10); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)


# Fig 1: diagonal
fig, axs = plt.subplots(2, 3, figsize=(17, 9))
for k, ti in enumerate(range(6)):
    panel(axs.flat[k], ti, ti)
axs.flat[0].legend(fontsize=11)
fig.suptitle("mh104 Timoshenko DIAGONAL vs f  (JAX-Kirchhoff shell OML/center vs FEniCS solid)", fontsize=15)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "enlarged_diagonal.png"), dpi=150, bbox_inches="tight")

# off-diagonal terms (i>j), split into two enlarged figures
od = [(i, j) for i in range(6) for j in range(i)]
for part, (lo, hi, nm) in enumerate([(0, 8, "enlarged_coupling_1"), (8, 15, "enlarged_coupling_2")]):
    sub = od[lo:hi]
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for k, (i, j) in enumerate(sub):
        panel(axs.flat[k], i, j)
    for k in range(len(sub), 8):
        axs.flat[k].axis("off")
    axs.flat[0].legend(fontsize=10)
    fig.suptitle("mh104 Timoshenko OFF-DIAGONAL coupling vs f  (part %d/2, OML/center vs solid)" % (part + 1), fontsize=15)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, nm + ".png"), dpi=150, bbox_inches="tight")

# Fig: %error diagonal (separate)
fig, axs = plt.subplots(2, 3, figsize=(17, 9))
for k, ti in enumerate(range(6)):
    ax = axs.flat[k]
    for r in REFS:
        y = [100 * (shell[(fi, r)][ti, ti] - solid[fi][ti, ti]) / abs(solid[fi][ti, ti])
             if (solid[fi] is not None and shell[(fi, r)] is not None) else np.nan for fi in FS]
        ax.plot(fv, y, "-o", color=col[r], ms=5, lw=1.8, label=r)
    ax.axhline(0, color="k", lw=0.8); ax.axhspan(-5, 5, color="0.85", alpha=0.5)
    ax.set_title(lab[ti], fontsize=13, fontweight="bold"); ax.set_xlabel("thickness factor f", fontsize=10)
    ax.set_ylabel("% error vs solid", fontsize=10); ax.grid(alpha=0.3)
axs.flat[0].legend(fontsize=11, title="shell ref")
fig.suptitle("mh104 shell vs solid: Timoshenko diagonal % ERROR (grey = +-5%, OML/center)", fontsize=15)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "pcterror_diagonal.png"), dpi=150, bbox_inches="tight")
print("wrote enlarged_diagonal.png, enlarged_coupling_1.png, enlarged_coupling_2.png, pcterror_diagonal.png")
