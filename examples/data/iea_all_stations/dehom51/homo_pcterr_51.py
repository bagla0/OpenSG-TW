"""homo_pcterr_51.py -- consolidated 51-station homogenization %error (RM shell 6x6 vs VABS .K),
one 2x3 panel per Timoshenko diagonal term, rainbow markers, +/-5% band.  For the RM paper."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
DAT = os.path.join(HERE, "out", "plots_rm_vs_vabs", "rm_vs_vabs_diag.dat")
OUT = "/home/roger/a/bagla0/claude_tmp/xsec_paper/figures/dehom_homo_pcterr_51.png"
d = np.loadtxt(DAT)
eta = d[:, 0]; err = d[:, 13:19]
LBL = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]
TIT = ["extension $EA$", "transv. shear $GA_2$", "transv. shear $GA_3$",
       "torsion $GJ$", "flap bending $EI_2$", "edge bending $EI_3$"]
col = plt.cm.rainbow(np.linspace(0, 1, 6))
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14})
fig, axs = plt.subplots(3, 2, figsize=(12, 13.5))            # 2 panels per row (enlarged)
for k in range(6):
    ax = axs.flat[k]
    ax.axhspan(-5, 5, color="0.9", zorder=0); ax.axhline(0, color="0.6", lw=1.2, ls=":")
    ax.plot(eta, err[:, k], "-o", color=col[k], mec="k", mew=0.5, ms=8, lw=2.2)
    mx = np.nanmax(np.abs(err[:, k])); ax.set_ylim(-max(6, 1.2 * mx), max(6, 1.2 * mx))
    ax.set_xlabel(r"span $r/R$"); ax.set_ylabel(r"$%s$  RM vs VABS  [\%%]" % LBL[k])
    ax.set_title(TIT[k], fontsize=15)                     # plain title (removed the colored in-plot text)
    ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT, dpi=150); plt.close(fig)
print("wrote", OUT)
for k in range(6):
    print("  %-4s mean %5.2f%%  max %6.2f%%" % (LBL[k], np.nanmean(np.abs(err[:, k])), np.nanmax(np.abs(err[:, k]))))
