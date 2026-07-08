"""plot_paper_convergence.py -- convergence figure for the tapered paper.

2x2 panels (rows: iso / m45, cols: circle / square): GA3 and GA2 %err vs hoop
divisions NC, eliminated vs constrained, fixed 3-D solid reference.  Also emits a
timing summary table (dat) from the same sweep.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "taper_indep_study")
FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "fig_convergence.png")
b = np.load(os.path.join(OUT, "paper_convergence.npz"), allow_pickle=True)
LEVELS = [(24, 5), (36, 8), (48, 10), (72, 15), (96, 20)]


def err(geom, mat, which, i):
    e = []
    for (nc, nl) in LEVELS:
        k = "%s_%s_nc%d_nl%d" % (geom, mat, nc, nl)
        if k + "_" + which in b.files and k + "_solid" in b.files:
            S = b[k + "_" + which]; So = b[k + "_solid"]
            e.append(100 * (S[i, i] - So[i, i]) / So[i, i])
        else:
            e.append(np.nan)
    return np.array(e)


ncs = np.array([nc for nc, _ in LEVELS])
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
for r, mat in enumerate(("iso", "m45")):
    for c, geom in enumerate(("circle", "square")):
        ax = axs[r, c]
        ax.plot(ncs, err(geom, mat, "ind", 2), "s-", color="#2a6", label="$C_{33}$ (GA$_3$)")
        ax.plot(ncs, err(geom, mat, "ind", 1), "o--", color="#26c", label="$C_{22}$ (GA$_2$)")
        ax.plot(ncs, err(geom, mat, "ind", 3), "^:", color="#a52", label="$C_{44}$ (GJ)")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_title("%s, %s" % (geom, "isotropic" if mat == "iso" else "$[-45^\\circ]$"), fontsize=11)
        ax.grid(alpha=0.25)
        if r == 1:
            ax.set_xlabel("hoop divisions $N_C$ (axial $\\propto N_C$)")
        if c == 0:
            ax.set_ylabel("error vs 3-D solid [%]")
axs[0, 0].legend(fontsize=9, loc="lower left", framealpha=0.9)
fig.suptitle("Shear and torsion convergence, strong taper $a_R=0.7$, thin wall $t/R=0.02$", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(FIG, dpi=170)
print("wrote", FIG)

# timing summary (all-6DOF: rings + segment breakdown)
with open(os.path.join(OUT, "timing_summary.dat"), "w") as f:
    f.write("# geom mat NC NL  t_rings_s  t_seg_s  t_total_s\n")
    for geom in ("circle", "square"):
        for mat in ("iso", "m45"):
            for (nc, nl) in LEVELS:
                k = "%s_%s_nc%d_nl%d" % (geom, mat, nc, nl)
                tr = float(b[k + "_trings"]) if k + "_trings" in b.files else np.nan
                tsg = float(b[k + "_tseg"]) if k + "_tseg" in b.files else np.nan
                f.write("%s %s %d %d %.1f %.1f %.1f\n" % (geom, mat, nc, nl, tr, tsg, tr + tsg))
print("wrote", os.path.join(OUT, "timing_summary.dat"))
