"""cpb_paths_fig.py -- the r=0.2 section-with-paths figure, now with THREE paths:
circumferential LP-OML loop, spar-cap through-thickness column, and the NEW
centre-web/cap T-JUNCTION column (x2 ~ 0.16, through the cap wall into the upper web).
Variant of dehom51/make_section_paths_fig.py; output figures/r020_section_paths3.png."""
import json
import os

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
D51 = os.path.abspath(os.path.join(HERE, "..", ".."))
IEA = os.path.abspath(os.path.join(D51, ".."))
SOL = os.path.join(IEA, "shell51", "2d_yaml", "iea_s10_solid.yaml")
CO = os.path.join(D51, "coords")


def frow(v):
    if isinstance(v, str):
        return [float(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [float(x) for x in v[0].split()]
    return [float(x) for x in v]


def irow(v):
    if isinstance(v, str):
        return [int(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [int(x) for x in v[0].split()]
    return [int(x) for x in v]


y = yaml.safe_load(open(SOL))
xy = np.array([frow(n)[:2] for n in y["nodes"]])
conn = [irow(e) for e in y["elements"]]
base = min(min(c) for c in conn)
conn = [[n - base for n in c] for c in conn]
ne = len(conn)
matnames = [m["name"] for m in y["materials"]]
E1 = np.array([frow(m["E"])[0] for m in y["materials"]])
emat = np.zeros(ne, dtype=int)
setlist = y["sets"]["element"]
lab_min = min(min(s["labels"]) for s in setlist)
for s in setlist:
    mi = matnames.index(s["name"]) if s["name"] in matnames else 0
    for lab in s["labels"]:
        emat[lab - lab_min] = mi
mnf = os.path.join(CO, "material_names.json")
mnames = json.load(open(mnf)) if os.path.exists(mnf) else {}
realname = lambda mi: mnames.get(matnames[mi], matnames[mi])
carbon = int(np.argmax(E1))

circ = np.loadtxt(os.path.join(CO, "iea_s10.circumferential.coords"))[:, :2]
col = np.loadtxt(os.path.join(CO, "iea_s10.lp_sparcap_left_thickness.coords"))[:, :2]

# junction path: the ACTUAL two-leg polyline (written by cpb_r020_final.py from the
# OML ring geometry: OML node -> inward skin normal through the wall -> web mid-line)
JP = os.path.join(HERE, "data", "junction_polyline_oml.dat")
jverts = np.loadtxt(JP)

cmap = plt.cm.tab10
fig, ax = plt.subplots(figsize=(13, 5))
ax.add_collection(PolyCollection([xy[c] for c in conn],
                                 facecolors=[cmap(emat[k] % 10) for k in range(ne)],
                                 edgecolors="none"))
handles = [Patch(facecolor=cmap(mi % 10),
                 label="%s%s" % (realname(mi), "  (carbon)" if mi == carbon else ""))
           for mi in range(len(matnames))]

ax.plot(circ[:, 0], circ[:, 1], "-", color="k", lw=2.6,
        label="circumferential path (LE $\\rightarrow$ TE)")
ax.plot(col[:, 0], col[:, 1], "-", color="cyan", lw=3.4,
        label="through-thickness path (outer $\\rightarrow$ inner)")
ax.plot(jverts[:, 0], jverts[:, 1], "-", color="magenta", lw=3.4,
        label="junction path (outer surface $\\rightarrow$ web)")
i0 = int(0.40 * len(circ))
d = circ[min(i0 + 3, len(circ) - 1)] - circ[i0 - 3]
d = d / np.linalg.norm(d)
ax.annotate("", xy=circ[i0] + 0.45 * d, xytext=circ[i0],
            arrowprops=dict(arrowstyle="-|>", color="k", lw=2.6, mutation_scale=28))
dt = col[-1] - col[0]
dt = dt / np.linalg.norm(dt)
ax.annotate("", xy=col[-1] + 0.16 * dt, xytext=col[0] - 0.30 * dt,
            arrowprops=dict(arrowstyle="-|>", color="cyan", lw=3.0, mutation_scale=30))
# single direction arrow along the WEB leg (the direction of travel at the path end)
dj_out = jverts[2] - jverts[1]
dj_out = dj_out / np.linalg.norm(dj_out)
ax.annotate("", xy=jverts[2] + 0.10 * dj_out, xytext=jverts[2] - 0.20 * dj_out,
            arrowprops=dict(arrowstyle="-|>", color="magenta", lw=3.0, mutation_scale=30))
ax.plot([0], [0], "+", color="k", mew=2.0, ms=14)
ax.set_aspect("equal")
ax.autoscale()
ax.axis("off")
lg1 = ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.add_artist(lg1)
ax.legend(handles=handles, loc="upper right", fontsize=10, title="materials",
          framealpha=0.95)
fig.tight_layout()
out = os.path.join(HERE, "figures", "r020_section_paths3.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
