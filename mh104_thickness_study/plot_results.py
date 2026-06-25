"""Plot the mh104 wall-thickness sweep: Timoshenko stiffnesses vs thickness factor for the 2D SOLID
(reference) and the 1D SHELL at OML (frac 0, priority) and CENTER (frac 0.5) references.  Shows the
thin-wall convergence: shell -> solid as the walls thin (f->0.2)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)
FACTORS = [0.2, 0.4, 0.6, 0.8, 1.0]
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def tag(f):
    return "f%03d" % int(round(f * 100))


def load(kind, f):
    p = os.path.join(RES, "C6_%s_%s.txt" % (kind, tag(f)))
    return np.loadtxt(p) if os.path.exists(p) else None


solid = {f: load("solid", f) for f in FACTORS}
oml = {f: load("shell_oml", f) for f in FACTORS}
cen = {f: load("shell_center", f) for f in FACTORS}
fok = [f for f in FACTORS if solid[f] is not None and oml[f] is not None]
print("factors with data:", fok)

# ---- Fig 1: stiffness vs thickness (solid, shell-OML, shell-center) ----
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, ax in enumerate(axes.flat):
    xs = fok
    ys_solid = [solid[f][i, i] for f in xs]
    ys_oml = [oml[f][i, i] for f in xs]
    ys_cen = [cen[f][i, i] if cen[f] is not None else np.nan for f in xs]
    ax.plot(xs, ys_solid, "ks-", lw=2, ms=7, label="2D solid (= VABS)")
    ax.plot(xs, ys_oml, "o-", color="tab:red", lw=2, ms=6, label="shell OML (frac 0)")
    ax.plot(xs, ys_cen, "^--", color="tab:blue", lw=1.8, ms=6, label="shell center (frac 0.5)")
    ax.set_title(LBL[i]); ax.set_xlabel("thickness factor f"); ax.grid(alpha=0.3)
    ax.set_ylabel("stiffness")
    if i == 0:
        ax.legend(fontsize=9)
fig.suptitle("mh104 Timoshenko stiffness vs wall-thickness factor — 1D shell vs 2D solid (=VABS)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(PLOTS, "timo_vs_thickness.png"), dpi=140)
print("wrote timo_vs_thickness.png")

# ---- Fig 2: % error (shell vs solid) vs thickness -- thin-wall convergence ----
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, ax in enumerate(axes.flat):
    xs = fok
    e_oml = [100 * (oml[f][i, i] - solid[f][i, i]) / solid[f][i, i] for f in xs]
    e_cen = [100 * (cen[f][i, i] - solid[f][i, i]) / solid[f][i, i] if cen[f] is not None else np.nan for f in xs]
    ax.plot(xs, e_oml, "o-", color="tab:red", lw=2, ms=6, label="shell OML")
    ax.plot(xs, e_cen, "^--", color="tab:blue", lw=1.8, ms=6, label="shell center")
    ax.axhline(0, color="k", lw=1)
    ax.set_title("%s  (shell - solid)/solid" % LBL[i]); ax.set_xlabel("thickness factor f")
    ax.set_ylabel("% error vs solid"); ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=9)
fig.suptitle("mh104 thin-wall convergence: shell % error vs 2D solid shrinks as walls thin (f->0.2)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(PLOTS, "pcterr_vs_thickness.png"), dpi=140)
print("wrote pcterr_vs_thickness.png")

# ---- table ----
print("\n%-4s %-7s | %-10s %-10s %-10s | OML%%err center%%err" % ("term", "f", "solid", "shellOML", "shellCen"))
with open(os.path.join(RES, "summary_table.txt"), "w") as fo:
    fo.write("term f solid shellOML shellCenter OML%err center%err\n")
    for i in range(6):
        for f in fok:
            s, o = solid[f][i, i], oml[f][i, i]
            c = cen[f][i, i] if cen[f] is not None else np.nan
            eo = 100 * (o - s) / s
            ec = 100 * (c - s) / s
            line = "%-4s %.2f %.4e %.4e %.4e %+.1f %+.1f" % (LBL[i], f, s, o, c, eo, ec)
            print("%-4s %.2f    | %.3e  %.3e  %.3e | %+6.1f   %+6.1f" % (LBL[i], f, s, o, c, eo, ec))
            fo.write(line + "\n")
print("\nwrote summary_table.txt")
