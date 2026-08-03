"""fig_blade_mesh.py -- the BAR-URC blade built from its 30 actual 1-D SG station contours, plus the
cross-section meshes at selected stations.  Renders the REAL nodes/elements from the yamls, never a sketch.

Left: 3-D blade -- every station contour drawn at its own z = k*100/29, with longitudinal lines joining
corresponding nodes, the carbon spar cap highlighted, and the governing station marked.
Right: the cross-section mesh at four stations, showing how the section and its layup evolve outboard.
"""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SHELLD = os.path.join(ROOT, "tests", "data")
NST, BLADE = 30, 100.0
GOV = 6
CAPMAT = "carbon_uni_industry_baseline"


def _row(x):
    out = []
    for v in ([x] if isinstance(x, str) else x):
        if isinstance(v, str):
            out.extend(float(t) for t in v.split())
        else:
            out.append(float(v))
    return out


def load(k):
    d = yaml.safe_load(open(os.path.join(SHELLD, "1Dshell_%d.yaml" % k)))
    nd = np.array([_row(n)[:2] for n in d["nodes"]])
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]]); cells -= cells.min()
    name_of = {}
    for g in d["sets"]["element"]:
        for lab in g["labels"]:
            name_of[int(lab) - 1] = g["name"]
    names = [name_of.get(i, d["sections"][0]["elementSet"]) for i in range(len(cells))]
    lay = {s["elementSet"]: s["layup"] for s in d["sections"]}
    # a "cap" element is one whose layup contains the carbon uni ply
    iscap = np.array([any(p[0] == CAPMAT for p in lay[nm]) for nm in names])
    thick = np.array([sum(float(p[1]) for p in lay[nm]) for nm in names])
    return nd, cells, np.array(names), iscap, thick


S = {k: load(k) for k in range(NST)}

fig = plt.figure(figsize=(15.5, 7.2))
ax = fig.add_subplot(1, 2, 1, projection="3d")

for k in range(NST):
    nd, cells, names, iscap, _ = S[k]
    z = k * BLADE / (NST - 1)
    for e, (a, b) in enumerate(cells):
        ax.plot([nd[a, 0], nd[b, 0]], [z, z], [nd[a, 1], nd[b, 1]],
                color="#D55E00" if iscap[e] else "#9ecae1",
                lw=1.7 if iscap[e] else 0.5, alpha=0.95 if iscap[e] else 0.55, zorder=3 if iscap[e] else 1)
# longitudinal lines: only where consecutive stations share a node count (webs appear at k=3)
for k in range(NST - 1):
    n0, n1 = S[k][0], S[k + 1][0]
    if len(n0) != len(n1):
        continue
    z0, z1 = k * BLADE / (NST - 1), (k + 1) * BLADE / (NST - 1)
    for j in range(0, len(n0), 3):
        ax.plot([n0[j, 0], n1[j, 0]], [z0, z1], [n0[j, 1], n1[j, 1]], color="0.6", lw=0.35, alpha=0.5)
# draw the governing station through its ELEMENTS -- the node ordering is not sequential around the
# contour ([1 2], [3 1], [2 4], ...), so joining nodes by index would zigzag across the section
ndg, cellsg = S[GOV][0], S[GOV][1]
zg = GOV * BLADE / (NST - 1)
for a, b in cellsg:
    ax.plot([ndg[a, 0], ndg[b, 0]], [zg, zg], [ndg[a, 1], ndg[b, 1]], color="k", lw=1.7, zorder=6)
ax.set_xlabel(r"$y_2$ [m]"); ax.set_ylabel("span $z$ [m]"); ax.set_zlabel(r"$y_3$ [m]")
ax.set_box_aspect((1.1, 4.2, 1.1)); ax.view_init(elev=20, azim=-62)
ax.legend(handles=[Line2D([], [], color="#D55E00", lw=2, label="carbon spar cap"),
                   Line2D([], [], color="#9ecae1", lw=1.2, label="skin / webs"),
                   Line2D([], [], color="k", lw=1.6, label="governing station 6")],
          loc="upper left", bbox_to_anchor=(0.0, 0.92), frameon=False, fontsize=9)

# ---- cross-section meshes at four stations ----
SHOW = [3, 6, 15, 25]
# one common frame for all four, so the sections are visually comparable and none is clipped
_al = np.vstack([S[k][0] for k in SHOW])
XL = (_al[:, 0].min() - 0.2, _al[:, 0].max() + 0.2)
YL = (_al[:, 1].min() - 0.15, _al[:, 1].max() + 0.15)
for i, k in enumerate(SHOW):
    a2 = fig.add_subplot(4, 2, 2 * (i + 1))
    nd, cells, names, iscap, thick = S[k]
    segs = np.array([[nd[p], nd[q]] for p, q in cells])
    lw = 1.0 + 5.0 * thick / thick.max()
    a2.add_collection(LineCollection(segs, colors=["#D55E00" if c else "#4292c6" for c in iscap],
                                     linewidths=lw))
    a2.plot(nd[:, 0], nd[:, 1], "k.", ms=1.8, zorder=5)
    a2.set_aspect("equal"); a2.autoscale_view()
    a2.set_xlim(*XL); a2.set_ylim(*YL)
    a2.text(0.01, 0.94, "st%02d   $z$=%.1f m   $r/R$=%.2f   %d nodes"
            % (k, k * BLADE / (NST - 1), k / (NST - 1.0), len(nd)),
            transform=a2.transAxes, va="top", fontsize=8.5)
    a2.set_xticks([]) if i < len(SHOW) - 1 else a2.set_xlabel(r"$y_2$ [m]")
    a2.set_yticks([])

fig.tight_layout()
out = os.path.join(HERE, "bar_urc_blade_mesh.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote %s (%.0f kB)" % (out, os.path.getsize(out) / 1024))
for k in (0, 3, 6, 15, 29):
    nd, cells, _, iscap, thick = S[k]
    print("   st%02d  nodes=%2d elems=%2d  cap elems=%2d  t %.1f-%.1f mm"
          % (k, len(nd), len(cells), int(iscap.sum()), 1e3 * thick.min(), 1e3 * thick.max()))
