"""make_figs.py -- journal-quality cross-section mesh figures for the RM cross-section
paper.  For IEA-22 r=0.2 and r=0.3:
  * SOLID 2-D cross-section, coloured by MATERIAL (ply layers resolved through the wall),
    face-on, white background, material legend;
  * SHELL 1-D ring, coloured by LAYUP region, with one e2 (blue) / e3 (black) arrow per
    region (OML->IML normal), matching the house orientation-plot convention.
Also the single-cell and two-cell tube shell rings.  Clean, captioned in LaTeX (no
in-figure titles).

    python make_figs.py
"""
import os
import sys

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TAPER = os.path.abspath(os.path.join(HERE, "..", "..", "taper"))
TAPERJAX = os.path.abspath(os.path.join(HERE, "..", "..", "taper_jax"))
TWP = os.path.abspath(os.path.join(HERE, ".."))
for q in (MITC, REPO, TAPER, TAPERJAX, os.path.expanduser("~/OpenSG_io")):
    sys.path.insert(0, q)

from xsec_5v6_master import _row
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, extract_boundary_submesh,
                                           _PERM3, _PERMF)

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
# colour-blind-safe qualitative palette (Okabe-Ito-ish), stable material->colour
PAL = np.array([[0.35, 0.35, 0.35], [0.00, 0.45, 0.70], [0.84, 0.37, 0.00],
                [0.00, 0.62, 0.45], [0.80, 0.47, 0.65], [0.90, 0.62, 0.00],
                [0.34, 0.71, 0.91], [0.60, 0.60, 0.60]])
PRETTY = {"gelcoat": "gelcoat", "glass_triax": "glass triax", "glass_biax": "glass biax",
          "glass_ud": "glass UD", "carbon_ud": "carbon UD", "medium_density_foam": "foam core",
          "resin": "resin", "adhesive": "adhesive"}


def _prettymat(m):
    return PRETTY.get(m, m.replace("_", " "))


# ------------------------------------------------------------------ SOLID cross-sections
def solid_boundaries():
    """Extract the r=0.2 (L) and r=0.3 (R) 2-D solid cross-sections from the mixed
    hex+tet segment; return {tag: (nodes2d(x2,x3), faces list, mat names per face, matset)}."""
    blade = load_blade(WINDIO)
    cs1 = build_cross_section(blade, 0.2, mesh_size=0.02)
    cs2 = build_cross_section(blade, 0.3, mesh_size=0.02)
    z1, z2 = blade_span_z(blade, 0.2), blade_span_z(blade, 0.3)
    res = hex_between_sections(cs1, cs2, z1, z2, nr=4, nsp=12, nw=3, mesh_size=0.02)
    oris, hmats = solid_yaml_payload(res, cs1, cs2)
    web = np.array([t[0] == "web" for t in res["htag"]])
    names = sorted(set(hmats)); nix = {n: i for i, n in enumerate(names)}
    seg = dict(nodes=np.asarray(res["nodes"])[:, _PERM3],
               batches={"hex8": (np.asarray(res["hexes"]),
                                 np.array([nix[m] for m in hmats], int),
                                 np.asarray(oris)[:, _PERMF])},
               mat_param=np.zeros((len(names), 9)), nelem=len(res["hexes"]))
    seg = split_batches_to_tets(seg, mask=web)
    out = {}
    for tag, side in (("r020", "L"), ("r030", "R")):
        b = extract_boundary_submesh(seg, side)
        out[tag] = (b, names)
    return out, names


def render_solid(tag, b, names, png):
    """b['nodes'] beam-first (beam,x2,x3); plot (x2,x3) face-on, colour by material."""
    P = np.asarray(b["nodes"])[:, [1, 2]]
    cells, cd = [], []
    for kind, (conn, mid, _f) in b["batches"].items():
        nn = 4 if kind == "quad4" else 3
        for row, m in zip(conn, mid):
            cells.append(row); cd.append(int(m))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    used = sorted(set(cd))
    from matplotlib.collections import PolyCollection
    polys = [P[c] for c in cells]
    colors = [PAL[m % len(PAL)] for m in cd]
    pc = PolyCollection(polys, facecolors=colors, edgecolors="k", linewidths=0.25)
    ax.add_collection(pc)
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    ax.legend(handles=[Patch(facecolor=PAL[m % len(PAL)], edgecolor="k", label=_prettymat(names[m]))
                       for m in used], loc="lower center", ncol=min(len(used), 4),
              fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(); fig.savefig(png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.basename(png), "(%d cells)" % len(cells), flush=True)


# ------------------------------------------------------------------ SHELL rings
def render_shell(shell_yaml, png, arrows=True):
    d = yaml.safe_load(open(shell_yaml))
    rx = np.array([_row(r)[:2] for r in d["nodes"]], float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], float)
    e2, e3 = ori[:, 3:5], ori[:, 6:8]
    sections = d["sections"]
    setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    sec = np.zeros(len(cells), int)
    for grp in d["sets"]["element"]:
        si = setname_to_sec[grp["name"]]
        for lab in grp["labels"]:
            sec[int(lab) - 1] = si
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for e, (a, bb) in enumerate(cells):
        ax.plot(rx[[a, bb], 0], rx[[a, bb], 1], "-", color=PAL[sec[e] % len(PAL)], lw=2.0)
    if arrows:
        cent = rx[cells].mean(1)
        L = 0.03 * (rx[:, 0].max() - rx[:, 0].min())
        step = max(1, len(cells) // 60)
        for e in range(0, len(cells), step):
            ax.arrow(*cent[e], L * e2[e, 0], L * e2[e, 1], color="tab:blue", width=L * 0.03,
                     head_width=L * 0.18, length_includes_head=True, zorder=3)
            ax.arrow(*cent[e], L * e3[e, 0], L * e3[e, 1], color="k", width=L * 0.03,
                     head_width=L * 0.18, length_includes_head=True, zorder=3)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.basename(png), "(%d cells, %d sections)" % (len(cells), len(sections)), flush=True)


if __name__ == "__main__":
    print("SOLID cross-sections (r=0.2, r=0.3) ...", flush=True)
    solids, names = solid_boundaries()
    for tag, (b, nm) in solids.items():
        render_solid(tag, b, nm, os.path.join(FIG, "iea_%s_solid.png" % tag))
    print("SHELL rings ...", flush=True)
    IB = os.path.join(TWP, "iea22_blade", "data")
    render_shell(os.path.join(IB, "shell_r020.yaml"), os.path.join(FIG, "iea_r020_shell.png"))
    render_shell(os.path.join(IB, "shell_r030.yaml"), os.path.join(FIG, "iea_r030_shell.png"))
    render_shell(os.path.join(TWP, "single_cell_tube", "data", "shell_center.yaml"),
                 os.path.join(FIG, "single_tube_shell.png"))
    render_shell(os.path.join(TWP, "two_cell_tube", "data", "tube2cell_thin.yaml"),
                 os.path.join(FIG, "two_cell_shell.png"))
    print("all figures ->", FIG)
