"""render_prevabs_solid.py -- IEA r=0.2 & r=0.3 PreVABS 2-D solid cross-sections rendered
FILLED by material with NO element edges (clean journal figure), + the 1-D shell rings.
Solid source = the PreVABS all-tri boundary yaml (OpenSG_io build_boundary --mesher prevabs).

    python render_prevabs_solid.py
"""
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "taper"))
from xsec_5v6_master import _row


def _real_name_map():
    """Map a material's E1 -> its windIO name (the PreVABS yaml uses generic Material_N)."""
    try:
        from taper_common import WINDIO
        from opensg_io.converter import load_blade
        b = load_blade(WINDIO)
        out = {}
        for nm, m in b.mats.items():
            e = m.get("E")
            e1 = (e[0] if isinstance(e, (list, tuple)) else e)
            if e1:
                out[round(float(e1) / 1e6)] = nm       # keyed by E1 in MPa (rounded)
        return out
    except Exception as ex:
        print("  (name map unavailable: %s)" % str(ex)[:60]); return {}


_E2NAME = _real_name_map()

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
PV = os.path.expanduser("~/OpenSG_io/examples/mesh_out/iea_prevabs_xsec")
PAL = np.array([[0.35, 0.35, 0.35], [0.00, 0.45, 0.70], [0.84, 0.37, 0.00],
                [0.00, 0.62, 0.45], [0.80, 0.47, 0.65], [0.90, 0.62, 0.00],
                [0.34, 0.71, 0.91], [0.60, 0.60, 0.60]])
PRETTY = {"gelcoat": "gelcoat", "glass_triax": "glass triax", "glass_biax": "glass biax",
          "glass_ud": "glass UD", "glass_uniax": "glass UD", "carbon_ud": "carbon UD",
          "carbon_uniax": "carbon UD", "medium_density_foam": "foam core", "foam": "foam core",
          "resin": "resin", "adhesive": "adhesive"}


def _pretty(m):
    return PRETTY.get(m, m.replace("_", " "))


def render_solid_noedges(yaml_path, png):
    d = yaml.safe_load(open(yaml_path))
    P = np.array([_row(r)[:2] for r in d["nodes"]], float)
    cells = [[int(v) - 1 for v in _row(e)] for e in d["elements"]]
    # recover real material names from E1 (PreVABS yaml labels them Material_N)
    mat_E1 = {}
    for m in d.get("materials", []):
        e = m.get("E"); e1 = e[0] if isinstance(e, (list, tuple)) else e
        mat_E1[m["name"]] = _E2NAME.get(round(float(e1) / 1e6), m["name"])
    names = [mat_E1.get(s["name"], s["name"]) for s in d["sets"]["element"]]
    mat_of = np.zeros(len(cells), int)
    for si, s in enumerate(d["sets"]["element"]):
        for lab in s["labels"]:
            mat_of[lab - 1] = si
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    polys = [P[c] for c in cells]
    cols = [PAL[m % len(PAL)] for m in mat_of]
    pc = PolyCollection(polys, facecolors=cols, edgecolors="none", antialiased=False)  # NO edges
    ax.add_collection(pc)
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    used = sorted(set(mat_of.tolist()))
    ax.legend(handles=[Patch(facecolor=PAL[m % len(PAL)], label=_pretty(names[m])) for m in used],
              loc="lower center", ncol=min(len(used), 5), fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(); fig.savefig(png, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.basename(png), "(%d tri, %d materials)" % (len(cells), len(used)), flush=True)


if __name__ == "__main__":
    for tag in ("r020", "r030"):
        p = os.path.join(PV, "%s_solid_boundary.yaml" % tag)
        if os.path.exists(p):
            render_solid_noedges(p, os.path.join(FIG, "iea_%s_solid_prevabs.png" % tag))
        else:
            print("  MISSING", p)
