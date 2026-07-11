"""make_plots.py -- convergence + thickness-sweep figures from the saved npz."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
LBL = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]
MARK = ["o", "s", "^", "D", "v", "P"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]


def conv_plot():
    d = np.load(os.path.join(RES, "conv_single_tube.npz"))
    N, e6 = d["N"], d["err6"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.axhline(0, color="0.6", lw=0.8)
    for k in range(6):
        ax.plot(N, e6[:, k], color=COL[k], marker=MARK[k], lw=1.6, ms=5, label="$%s$" % LBL[k])
    ax.set_xscale("log"); ax.set_xlabel("circumferential elements $N$")
    ax.set_ylabel("diagonal % error vs 2-D solid")
    ax.grid(alpha=0.3, which="both"); ax.legend(ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(); p = os.path.join(FIG, "conv_single_tube.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", p)


def tube_sweep_plot():
    d = np.load(os.path.join(RES, "tube_thick_sweep.npz"))
    tR, e6 = d["tR"], d["err6"]
    order = np.argsort(tR)
    tR, e6 = tR[order], e6[order]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.axhline(0, color="0.6", lw=0.8)
    for k in range(6):
        ax.plot(tR, e6[:, k], color=COL[k], marker=MARK[k], lw=1.6, ms=5, label="$%s$" % LBL[k])
    ax.set_xlabel("wall thickness ratio  $t/R$   (thin $\\rightarrow$ thick)")
    ax.set_ylabel("diagonal % error vs 2-D solid")
    ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(); p = os.path.join(FIG, "tube_thick_sweep.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    conv_plot()
    if os.path.exists(os.path.join(RES, "tube_thick_sweep.npz")):
        tube_sweep_plot()
