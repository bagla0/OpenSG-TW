"""Compare layup orientation (e1,e2,e3) between the SOLID 2D mesh (VABS-validated, 3-node tris) and
the SHELL 1D mesh (2-node lines).  Both YAMLs store elementOrientations as 9 components [e1,e2,e3].

Quantitative check: for every shell element, find the nearest solid element centroid and compute the
in-plane dot products e2.e2 and e3.e3.  +1 => same direction (layup matches); -1 => flipped.
Visual: solid vs shell e2 and e3 arrow fields on the same axes, plus web/LE zooms.

Usage: python compare_orient_solid_shell.py <solid.yaml> <shell.yaml> [out_prefix]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

SOLID = sys.argv[1]
SHELL = sys.argv[2]
PRE = sys.argv[3] if len(sys.argv) > 3 else "cmp_solid_shell"


def _row(r):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)]


def _irow(r):
    return [int(float(v)) for v in (str(r[0]).split() if len(r) == 1 else r)]


def load(path):
    d = yaml.safe_load(open(path))
    nodes = np.array([_row(r) for r in d["nodes"]])
    elems = [np.array(_irow(r)) - 1 for r in d["elements"]]    # 0-based, variable length
    oris = np.array([_row(r) for r in d["elementOrientations"]])
    ctr = np.array([nodes[e, :2].mean(0) for e in elems])
    return dict(nodes=nodes, elems=elems, ctr=ctr,
                e1=oris[:, 0:3], e2=oris[:, 3:6], e3=oris[:, 6:9])


S = load(SOLID)   # solid (reference)
H = load(SHELL)   # shell

# ---- quantitative: nearest solid element to each shell element ----
d2 = ((H["ctr"][:, None, :] - S["ctr"][None, :, :]) ** 2).sum(-1)
nn = d2.argmin(1)
dist = np.sqrt(d2[np.arange(len(nn)), nn])
e2dot = (H["e2"][:, :2] * S["e2"][nn, :2]).sum(1)
e3dot = (H["e3"][:, :2] * S["e3"][nn, :2]).sum(1)
e1dot = (H["e1"] * S["e1"][nn]).sum(1)

print("shell elems=%d  solid elems=%d" % (len(H["ctr"]), len(S["ctr"])))
print("nearest-solid distance: median=%.4f  max=%.4f" % (np.median(dist), dist.max()))
print("e1.e1 : min=%.4f  (expect +1 everywhere)" % e1dot.min())
print("e2.e2 : mean=%+.3f  min=%+.3f   |e2.e2|>0.9: %d/%d   (<0 i.e. flipped: %d)"
      % (e2dot.mean(), e2dot.min(), (np.abs(e2dot) > 0.9).sum(), len(e2dot), (e2dot < 0).sum()))
print("e3.e3 : mean=%+.3f  min=%+.3f   |e3.e3|>0.9: %d/%d   (<0 i.e. flipped: %d)"
      % (e3dot.mean(), e3dot.min(), (np.abs(e3dot) > 0.9).sum(), len(e3dot), (e3dot < 0).sum()))
flip3 = np.where(e3dot < 0)[0]
if len(flip3):
    print("  e3-FLIPPED shell elems (x,y, e3dot):")
    for i in flip3[:20]:
        print("    elem %3d  (%.3f, %.3f)  e3dot=%+.3f  dist=%.4f" % (i, H["ctr"][i, 0], H["ctr"][i, 1], e3dot[i], dist[i]))

# ---- visual ----
alen = 0.045


def field(ax, M, which, title, sub=1, zoom=None, color="C0"):
    vec = M[which]
    c = M["ctr"][::sub]
    ax.quiver(c[:, 0], c[:, 1], vec[::sub, 0], vec[::sub, 1], color=color,
              angles="xy", scale_units="xy", scale=1.0 / alen, width=0.0026, zorder=3)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])


sub = max(1, len(S["ctr"]) // 700)   # subsample solid arrows to ~700
fig, axs = plt.subplots(4, 1, figsize=(13, 13))
field(axs[0], S, "e2", "SOLID  e2 (in-plane tangent / fiber)", sub=sub, color="C3")
field(axs[1], H, "e2", "SHELL  e2 (in-plane tangent)", color="C0")
field(axs[2], S, "e3", "SOLID  e3 (ply normal OML->IML)", sub=sub, color="C3")
field(axs[3], H, "e3", "SHELL  e3 (ply normal OML->IML)", color="C0")
fig.suptitle("Layup orientation: SOLID (2D, red) vs SHELL (1D, blue)", fontsize=12)
fig.tight_layout(); fig.savefig(PRE + "_full.png", dpi=140, bbox_inches="tight")
print("wrote", PRE + "_full.png")

# zooms: rear web region + LE region, e3, solid+shell overlaid
wx = 0.5 * (S["ctr"][:, 0].min() + S["ctr"][:, 0].max())
rear = 0.50    # rear web ~ x=0.5 (chord units, model frame)
zooms = {"rearweb": (rear - 0.12, rear + 0.12, -0.13, 0.13),
         "LE": (S["ctr"][:, 0].min() - 0.02, S["ctr"][:, 0].min() + 0.18, -0.13, 0.13)}
fig2, axs2 = plt.subplots(2, 1, figsize=(9, 9))
for ax, (nm, zb) in zip(axs2, zooms.items()):
    field(ax, S, "e3", "", sub=1, zoom=zb, color="C3")
    field(ax, H, "e3", "e3 overlay @ %s  (SOLID red, SHELL blue)" % nm, sub=1, zoom=zb, color="C0")
fig2.tight_layout(); fig2.savefig(PRE + "_e3zoom.png", dpi=150, bbox_inches="tight")
print("wrote", PRE + "_e3zoom.png")
