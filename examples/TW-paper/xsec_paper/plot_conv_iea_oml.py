"""plot_conv_iea_oml.py -- Fig 9: IEA r/R=0.2 RM 6-DOF contour-refinement convergence.

Reads results/ex3_iea_conv.npz (from ex3_iea_conv_oml.py) and plots the six Timoshenko
%errors vs the contour NODE COUNT.  Conventions per reviewer request:
  * x-axis: the actual node counts as plain integer ticks (no 10^2 offset)
  * y-axis label: no 'diagonal' wording
Writes conv_iea_r020.png (overwrites the paper figure name).
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "results", "ex3_iea_conv.npz"))
nn = np.asarray(z["nnode"], float)
err = np.asarray(z["diag_err"])
LBL = [str(x) for x in z["labels"]]
# drop the 143-node point: it is only ~9 elements coarser than 132 and adds no
# information, cluttering the low end of the sweep (reviewer request).
keep = nn != 143
nn, err = nn[keep], err[keep]
TIT = [r"$EA$", r"$GA_2$", r"$GA_3$", r"$GJ$", r"$EI_2$", r"$EI_3$"]
col = plt.cm.rainbow(np.linspace(0, 1, 6))

plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "legend.fontsize": 13,
                     "xtick.labelsize": 13, "ytick.labelsize": 14})
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.axhspan(-5, 5, color="0.92", zorder=0)
ax.axhline(0, color="0.6", lw=1.0, ls=":")
order = np.argsort(nn)
for k in range(6):
    ax.plot(nn[order], err[order, k], "-o", color=col[k], mec="k", mew=0.5,
            ms=7, lw=1.8, label=TIT[k])
ax.set_xlabel("number of contour nodes")
ax.set_ylabel(r"RM shell error vs.\ VABS  [\%]")
ax.set_xticks(np.sort(nn))                                # ticks AT the data points
ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax.ticklabel_format(axis="x", style="plain")
ax.set_xticklabels([("%d" % v) for v in np.sort(nn)], rotation=0)
ax.grid(alpha=0.25)
ax.legend(ncol=3, loc="best", frameon=False)
fig.tight_layout()
OUT = os.path.join(HERE, "figures", "conv_iea_r020.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("wrote", OUT, "node counts:", np.sort(nn).astype(int).tolist())
