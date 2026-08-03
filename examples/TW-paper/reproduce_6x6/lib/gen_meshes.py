"""Generate the 1D-shell tube mesh (single closed circle) on a chosen radius.

Pure numpy + yaml (NO JAX) so it runs anywhere.  Each mesh = n linear (2-node)
segments on a circle of radius R_ref:
  e1 = (0,0,1) beam axis;  e2 = CCW tangent;  e3 = inward radial (ply normal).
Nodes are placed DIRECTLY on the circle (not folded inward from the OML); the
reference plane is selected downstream (JAX: shift_abd_reference by d_shift;
FEniCS: compute_ABD(frac)).
"""
import os

import numpy as np
import yaml as _yaml

# ---- defaults (ud_frp orthotropic; overridden per call for the single-cell tube) ----
ANI = {"E": [37.0e9, 9.0e9, 9.0e9], "G": [4.0e9, 4.0e9, 4.0e9],
       "nu": [0.28, 0.28, 0.28]}
R_OUT = 1.0
H = 0.06
LAYUP = [(45.0, 0.03), (-45.0, 0.03)]
N = 160

REFS = {"OML":    (R_OUT,           0.0,     0.0),
        "center": (R_OUT - H / 2.0, H / 2.0, 0.5),
        "IML":    (R_OUT - H,       H,       1.0)}


class FlowList(list):
    pass


_yaml.add_representer(FlowList, lambda d, data: d.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=True))


def gen_tube_yaml(path, R_ref, layup=LAYUP, mat=ANI, n=N, ccw=True):
    """Write an OpenSG 1D-shell YAML for a circle of radius R_ref.

    ccw=True traverses counter-clockwise (matches the PreVABS solid baseline so
    the off-axis ply angle and the small extension-twist couplings agree).  e3 is
    ALWAYS the inward radial (OML->IML stacking normal), computed from geometry,
    so the layup stacks into the wall regardless of traversal."""
    s = 1.0 if ccw else -1.0
    ang = np.array([s * 2.0 * np.pi * k / n for k in range(n)])
    pts = [(R_ref * np.cos(t), R_ref * np.sin(t)) for t in ang]
    elems = [(k + 1, k + 2) for k in range(n - 1)] + [(n, 1)]
    ori = []
    for (a, b) in elems:
        p1 = np.array(pts[a - 1]); p2 = np.array(pts[b - 1])
        t = p2 - p1; e2 = t / (np.linalg.norm(t) + 1e-30)               # traversal tangent
        mid = 0.5 * (p1 + p2); e3 = -mid / (np.linalg.norm(mid) + 1e-30)  # inward radial
        ori.append([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0,
                    float(e3[0]), float(e3[1]), 0.0])
    data = {
        "nodes": [FlowList(["%.10f %.10f 0.0" % (x, y)]) for (x, y) in pts],
        "elements": [FlowList(["%d %d" % (a, b)]) for (a, b) in elems],
        "sets": {"element": [{"name": "tube", "labels": list(range(1, n + 1))}]},
        "sections": [{"type": "shell", "elementSet": "tube",
                      "layup": [["mat", float(t), float(a)] for a, t in layup]}],
        "materials": [{"name": "mat", "density": 1860.0, "elastic": mat}],
        "elementOrientations": [FlowList([float(v) for v in o]) for o in ori],
    }
    with open(path, "w") as f:
        _yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    return path


def gen_all(datadir, ccw=True):
    os.makedirs(datadir, exist_ok=True)
    out = {}
    for ref, (R_ref, _d, _f) in REFS.items():
        p = os.path.join(datadir, "shell_%s.yaml" % ref)
        gen_tube_yaml(p, R_ref, ccw=ccw)
        out[ref] = p
    return out
