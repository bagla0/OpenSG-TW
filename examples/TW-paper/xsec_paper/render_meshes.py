"""render_meshes.py -- clean cross-section mesh figures rendered DIRECTLY from the YAMLs
used for the homogenization: the 2-D solid mesh (filled by material, no element edges)
with the 1-D shell mid-surface contour overlaid as a DOTTED RED centre-reference line
(the ASC-paper Fig.-2 style).  Robust to triangle/quad elements and zero/duplicate
padding.  -> figures/twocell_mesh.png, figures/ell4cell_mesh.png
"""
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
TC = os.path.abspath(os.path.join(HERE, "..", "two_cell_tube", "data"))
ELM = os.path.join(HERE, "ellipse", "meshes")
FILL = ["#9ec7e8", "#f6b78b", "#a7d6a0", "#c8b3de", "#f0a3a3", "#d9c69a"]


def nodes_xy(d):
    out = []
    for n in d["nodes"]:
        v = n[0].split() if (isinstance(n, list) and n and isinstance(n[0], str)) else n
        out.append([float(v[0]), float(v[1])])
    return np.array(out)


def elem_ids(e):
    toks = e[0].split() if (isinstance(e, list) and e and isinstance(e[0], str)) else e
    ids = [int(round(float(x))) for x in toks]
    if ids and min(ids) >= 1:
        ids = [i - 1 for i in ids]                       # 1-based -> 0-based
    # strip padding: trailing repeats / negatives
    clean = []
    for i in ids:
        if i < 0:
            continue
        if clean and i == clean[-1]:
            continue
        clean.append(i)
    if len(clean) >= 4 and clean[-1] == clean[0]:
        clean = clean[:-1]
    return clean[:4] if len(clean) >= 4 else clean[:3]


def material_of(d, nelem):
    m = np.zeros(nelem, dtype=int)
    sets = d.get("sets", {}).get("element", [])
    for si, grp in enumerate(sets):
        for lab in grp["labels"]:
            m[int(lab) - 1] = si
    return m, [g["name"] for g in sets]


def render(solid_yaml, shell_yaml, out, title):
    ds = yaml.safe_load(open(solid_yaml))
    nd = nodes_xy(ds)
    polys, cidx = [], []
    mat, names = material_of(ds, len(ds["elements"]))
    for k, e in enumerate(ds["elements"]):
        ii = elem_ids(e)
        if len(ii) < 3:
            continue
        polys.append(nd[ii]); cidx.append(mat[k])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pc = PolyCollection(polys, facecolors=[FILL[c % len(FILL)] for c in cidx],
                        edgecolors="none", linewidths=0)
    ax.add_collection(pc)
    # dotted red centre-reference contour from the shell yaml
    dsh = yaml.safe_load(open(shell_yaml))
    ns = nodes_xy(dsh)
    segs = [elem_ids(e) for e in dsh["elements"]]
    for ii in segs:
        if len(ii) >= 2:
            ax.plot(ns[ii[:2], 0], ns[ii[:2], 1], color="#d62728", lw=1.4, ls=(0, (2, 2)))
    ax.plot([], [], color="#d62728", lw=1.4, ls=(0, (2, 2)), label="shell centre reference")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
    ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out), "| solid tris/quads:", len(polys),
          "| shell segs:", len(segs))


render(os.path.join(TC, "solid_tube2cell_thin.yaml"),
       os.path.join(TC, "tube2cell_thin.yaml"),
       os.path.join(FIG, "twocell_mesh.png"),
       "Webbed two-cell tube")
render(os.path.join(ELM, "solid_ell4cell_iso.yaml"),
       os.path.join(ELM, "shell_ell4cell_iso.yaml"),
       os.path.join(FIG, "ell4cell_mesh.png"),
       "Elliptic four-cell tube")
