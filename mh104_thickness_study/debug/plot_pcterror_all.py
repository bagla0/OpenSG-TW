"""Separate %error plot for ALL nonzero Timoshenko 6x6 terms (diagonal + off-diagonal coupling),
laid out as the 6x6 matrix.  OML + center vs solid.  Terms whose |solid| stays < 1% of EA are tagged
'(small)' and shaded -- their %error is numerically noisy.  y clipped to +-80%."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mh104_thickness_study"
RES = os.path.join(STUDY, "results"); FIG = os.path.join(STUDY, "figures")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 40, 60, 80, 100]; fv = [x / 100 for x in FS]
REFS = ["OML", "center"]; col = {"OML": "tab:blue", "center": "tab:purple"}


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


shell = {(fi, r): Ld(os.path.join(RES, "C6_shell_jax_%s_f%03d.txt" % (r, fi))) for fi in FS for r in REFS}
solid = {fi: Ld(os.path.join(RES, "C6_solid_f%03d.txt" % fi)) for fi in FS}
EAref = np.nanmean([solid[fi][0, 0] for fi in FS if solid[fi] is not None])

fig, axs = plt.subplots(6, 6, figsize=(22, 19))
for i in range(6):
    for j in range(6):
        ax = axs[i, j]
        if j > i:
            ax.axis("off"); continue
        small = all((solid[fi] is None) or (abs(solid[fi][i, j]) < 0.01 * solid[fi][0, 0]) for fi in FS)
        for r in REFS:
            y = [100 * (shell[(fi, r)][i, j] - solid[fi][i, j]) / abs(solid[fi][i, j])
                 if (solid[fi] is not None and shell[(fi, r)] is not None and abs(solid[fi][i, j]) > 0) else np.nan
                 for fi in FS]
            ax.plot(fv, y, "-o", color=col[r], ms=3, lw=1.3, label=r)
        ax.axhline(0, color="k", lw=0.7); ax.axhspan(-5, 5, color="0.85", alpha=0.6)
        ax.set_ylim(-80, 80)
        ttl = (lab[i] if i == j else "%s-%s" % (lab[i], lab[j])) + (" (small)" if small else "")
        ax.set_title(ttl, fontsize=9, fontweight="bold" if i == j else "normal",
                     color="0.5" if small else "k")
        if small:
            ax.set_facecolor("0.95")
        ax.tick_params(labelsize=6); ax.grid(alpha=0.2)
        if i == 5:
            ax.set_xlabel("f", fontsize=8)
axs[0, 0].legend(fontsize=9, loc="upper left")
fig.suptitle("mh104: % ERROR of ALL Timoshenko 6x6 terms vs solid (OML/center; grey band +-5%; "
             "'small' = |solid| < 1% EA, noisy)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(FIG, "pcterror_all_terms.png"), dpi=130, bbox_inches="tight")
print("wrote figures/pcterror_all_terms.png")
