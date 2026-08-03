"""fig_mode.py -- BAR-URC station 6 cross-section: actual contour, layup groups, web junctions, and the
buckling mode amplitude.  Renders the REAL mesh straight from the yaml/bundle (never a parametric sketch).

Conventions followed: no figure title (the LaTeX caption is the title); legend outside the axes, vertical,
on the right; equal aspect; colourblind-safe palette.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ST = int(os.environ.get("ST", "6"))
d = np.load(os.path.join(HERE, "mode_st%d.npz" % ST), allow_pickle=True)
nd, cells, amp, names, N = d["nodes"], d["cells"], d["amp"], d["names"], d["N"]
lam, mstar = d["lam"], int(d["mstar"])
DZ = 100.0 / 29.0

val = np.zeros(len(nd), int)
for a, b in cells:
    val[a] += 1; val[b] += 1
junc = np.where(val >= 3)[0]
segs = np.array([[nd[a], nd[b]] for a, b in cells])
CAP = "layup_6"

# stack vertically: the section is wide and flat, so side-by-side panels with equal aspect waste the frame
fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.5))

# ---- (a) layup groups: which wall is which ----
ax = axes[0]
uniq = sorted(set(names.tolist()))
cmap = plt.get_cmap("tab20")
col = {nm: cmap(i % 20) for i, nm in enumerate(uniq)}
ax.add_collection(LineCollection(segs, colors=[col[n] for n in names], linewidths=3.0))
ax.plot(nd[junc, 0], nd[junc, 1], "kv", ms=9, mfc="none", mew=2.0, zorder=5)
handles = [Line2D([], [], color=col[nm], lw=3,
                  label=nm + (" (spar cap)" if nm == CAP else "")) for nm in uniq]
handles.append(Line2D([], [], color="k", marker="v", ls="none", mfc="none", mew=2.0, label="web junction"))
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)

# ---- (b) pre-buckling N11 ----
ax = axes[1]
n11 = N[:, 0] / 1e6
lc = LineCollection(segs, cmap="coolwarm", linewidths=3.5)
lc.set_array(n11); lc.set_clim(-np.abs(n11).max(), np.abs(n11).max())
ax.add_collection(lc)
cb = fig.colorbar(lc, ax=ax, fraction=0.030, pad=0.02)
cb.set_label(r"$N_{11}$  [MN/m]   (negative = compression)", fontsize=9)

# ---- (c) mode amplitude ----
ax = axes[2]
lc2 = LineCollection(segs, cmap="viridis", linewidths=3.5)
lc2.set_array(amp); lc2.set_clim(0, 1)
ax.add_collection(lc2)
cb2 = fig.colorbar(lc2, ax=ax, fraction=0.030, pad=0.02)
cb2.set_label("normalised mode amplitude", fontsize=9)
cap = np.array([e for e, nm in enumerate(names) if nm == CAP])
mid = np.array([0.5 * (nd[cells[e, 0]] + nd[cells[e, 1]]) for e in cap])
ax.plot(mid[:, 0], mid[:, 1], "r.", ms=4, zorder=6)
for j in junc:
    ax.plot(nd[j, 0], nd[j, 1], "kv", ms=9, mfc="none", mew=2.0, zorder=6)

for ax, lab in zip(axes, ("(a) layup groups", "(b) pre-buckling $N_{11}$", "(c) mode amplitude")):
    ax.set_aspect("equal"); ax.autoscale_view()
    ax.set_xlabel(r"$y_2$  [m]"); ax.set_ylabel(r"$y_3$  [m]")
    ax.set_xlim(nd[:, 0].min() - 0.15, nd[:, 0].max() + 0.15)
    ax.set_ylim(nd[:, 1].min() - 0.15, nd[:, 1].max() + 0.15)
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, va="top", fontsize=10)

fig.tight_layout()
out = os.path.join(HERE, "bar_urc_st%02d_mode.png" % ST)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote %s  (%.0f kB)" % (out, os.path.getsize(out) / 1024))
print("station %d  z=%.2f m  r/R=%.3f   lam1=%.4f   m*=%d  half-wave=%.3f m"
      % (ST, ST * DZ, ST / 29.0, lam[0], mstar, DZ / mstar))
print("cap elements=%d   web junctions at nodes %s" % (len(cap), junc.tolist()))
