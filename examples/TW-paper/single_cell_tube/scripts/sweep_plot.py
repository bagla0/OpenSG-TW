"""R/h = 1..10 convergence plots (thesis Fig 3.6 style): relative error of the JAX
shell Timoshenko terms vs the FEniCS-2D-solid, center reference, k22=-1/R.
Two separate figures -- one for JAX-RM, one for JAX-KL.  Six diagonal terms
C11..C66 = [EA, GA2, GA3, GJ, EI2, EI3] (solid lines) plus the three dominant
couplings C14 (EA-GJ), C25 (GA2-EI2), C36 (GA3-EI3) (dashed).  Legend OUTSIDE the
axes (right) so it never covers the curves."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tests\research\tube_thesis_314\sweep\data"
FIGS = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tests\research\tube_thesis_314\sweep\figs"
os.makedirs(FIGS, exist_ok=True)
RH = list(range(1, 11))
LAB = [r"$C_{11}$", r"$C_{22}$", r"$C_{33}$", r"$C_{44}$", r"$C_{55}$", r"$C_{66}$"]
PHYS = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
COL = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
MK = ["o", "s", "^", "D", "v", "P"]
COUP = [((0, 3), r"$C_{14}$", "C14 EA-GJ"), ((1, 4), r"$C_{25}$", "C25 GA2-EI2"),
        ((2, 5), r"$C_{36}$", "C36 GA3-EI3")]
CCOL = ["#000000", "#7f7f7f", "#e377c2"]
CMK = ["X", "*", "d"]


def L(tag, rh):
    p = os.path.join(DATA, "C6_%s_rh%02d.txt" % (tag, rh))
    if not os.path.exists(p):
        return None
    M = np.loadtxt(p)
    return 0.5 * (M + M.T)


err = {"kirch": np.full((6, len(RH)), np.nan), "rm": np.full((6, len(RH)), np.nan)}
cerr = {"kirch": np.full((3, len(RH)), np.nan), "rm": np.full((3, len(RH)), np.nan)}
for k, rh in enumerate(RH):
    S = L("solid", rh)
    if S is None:
        continue
    for tag in ("kirch", "rm"):
        M = L("jax_%s" % tag, rh)
        if M is None:
            continue
        for i in range(6):
            err[tag][i, k] = 100.0 * (M[i, i] - S[i, i]) / S[i, i]
        for c, ((i, j), _, _) in enumerate(COUP):
            cerr[tag][c, k] = 100.0 * (M[i, j] - S[i, j]) / S[i, j]


def plot(tag, model, fname):
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for i in range(6):
        ax.plot(RH, err[tag][i], marker=MK[i], color=COL[i], lw=2.2, ms=9,
                mfc="none", mew=1.9, label=LAB[i])
    for c, ((i, j), lab, _) in enumerate(COUP):
        ax.plot(RH, cerr[tag][c], marker=CMK[c], color=CCOL[c], lw=2.0, ls="--",
                ms=10, mfc="none", mew=1.9, label=lab)
    ax.axhspan(-1.0, 1.0, color="tab:green", alpha=0.10, zorder=0)
    ax.axhline(0.0, color="0.35", lw=1.2)
    ax.set_xlabel(r"$R/h$", fontsize=20)
    ax.set_ylabel(r"Relative error  (%)", fontsize=19)
    ax.set_xticks(RH)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=14,
                    ncol=1, framealpha=0.95, title=model, title_fontsize=16,
                    handlelength=2.4, borderaxespad=0.0)
    leg._legend_box.align = "left"
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(fname))


plot("rm", "JAX-RM", os.path.join(FIGS, "sweep_RM.png"))
plot("kirch", "JAX-KL", os.path.join(FIGS, "sweep_KL.png"))

# error table (diagonal + couplings)
lines = ["R/h convergence: relative error (%) vs FEniCS-solid  (center ref, k22=-1/R, N=3200, [-45])", ""]
for tag, model in (("kirch", "JAX-KL"), ("rm", "JAX-RM")):
    lines.append("=== %s ===" % model)
    lines.append("R/h  " + "  ".join("%8s" % p for p in PHYS) + "  " +
                 "  ".join("%10s" % c[2].split()[0] for c in COUP))
    for k, rh in enumerate(RH):
        row = "%3d  " % rh + "  ".join("%+8.2f" % err[tag][i, k] for i in range(6))
        row += "  " + "  ".join("%+10.2f" % cerr[tag][c, k] for c in range(3))
        lines.append(row)
    lines.append("")
open(os.path.join(DATA, "sweep_errors.txt"), "w").write("\n".join(lines) + "\n")
print("wrote sweep_errors.txt")
