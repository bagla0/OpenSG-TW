"""Generate the SEPARATE 1D-shell tube meshes for the [45/-45] anisotropic tube
at the OML / center / IML reference radii.  Pure numpy + yaml (NO JAX) so it runs
under both the Windows JAX env and the WSL FEniCS env.

Each mesh = N linear (2-node) segments on a circle of radius R_ref:
  e1 = (0,0,1) beam axis;  e2 = CCW tangent;  e3 = inward radial (ply normal).
The nodes are placed DIRECTLY on each circle (R_out, R_mid, R_in) -- NOT folded
inward from the OML by offset_oml_to_iml.  The full [45/-45] layup is written on
every mesh; the reference plane is selected downstream (JAX: shift_abd_reference
by d_shift; FEniCS: compute_ABD(frac)).  See README.txt.
"""
import os
import numpy as np
import yaml as _yaml

# ---- geometry / material (matches cases/tube45.xml + the PreVABS solid wall) ----
ANI = {"E": [37.0e9, 9.0e9, 9.0e9], "G": [4.0e9, 4.0e9, 4.0e9],
       "nu": [0.28, 0.28, 0.28]}                 # ud_frp orthotropic
R_OUT = 1.0                                       # OML radius (XML circle baseline)
H = 0.06                                          # wall = [45/-45], 2 x 0.03
LAYUP = [(45.0, 0.03), (-45.0, 0.03)]             # OML-first: outer 45, inner -45
N = 160                                           # circumferential segments

# name -> (R_ref, d_shift, frac).  d_shift = JAX shift_abd_reference distance;
# frac = FEniCS compute_ABD reference fraction.  Solid wall [R_in,R_out] fixed.
REFS = {"OML":    (R_OUT,           0.0,   0.0),
        "center": (R_OUT - H / 2.0, H / 2.0, 0.5),
        "IML":    (R_OUT - H,       H,     1.0)}


class FlowList(list):
    pass


_yaml.add_representer(FlowList, lambda d, data: d.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=True))


def gen_tube_yaml(path, R_ref, layup=LAYUP, mat=ANI, n=N, ccw=True):
    """ccw=True (default) traverses the circle COUNTER-CLOCKWISE to match the
    tube45.xml circle baseline (<direction>ccw</direction>) and the PreVABS solid,
    so the element tangent e2 -- and therefore the off-axis ply angle and the
    small extension-twist couplings -- match the solid.  e3 is ALWAYS the inward
    radial (OML->IML stacking normal), computed from the geometry, so the layup
    stacks into the wall for both JAX and FEniCS regardless of traversal."""
    s = 1.0 if ccw else -1.0
    ang = np.array([s * 2.0 * np.pi * k / n for k in range(n)])
    pts = [(R_ref * np.cos(t), R_ref * np.sin(t)) for t in ang]
    elems = [(k + 1, k + 2) for k in range(n - 1)] + [(n, 1)]
    ori = []
    for (a, b) in elems:
        p1 = np.array(pts[a - 1]); p2 = np.array(pts[b - 1])
        t = p2 - p1; e2 = t / (np.linalg.norm(t) + 1e-30)   # traversal tangent (CW or CCW)
        mid = 0.5 * (p1 + p2); e3 = -mid / (np.linalg.norm(mid) + 1e-30)  # inward radial
        ori.append([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0,
                    float(e3[0]), float(e3[1]), 0.0])
    # FEniCS ShellBounMesh format: nodes/elements as space-separated strings,
    # sections type=shell, orientations a flat 9-float list (also read by JAX load_yaml).
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
    """Write shell_{OML,center,IML}.yaml into datadir (CCW to match tube45.xml).
    Returns {ref: path}."""
    os.makedirs(datadir, exist_ok=True)
    out = {}
    for ref, (R_ref, _d, _f) in REFS.items():
        p = os.path.join(datadir, "shell_%s.yaml" % ref)
        gen_tube_yaml(p, R_ref, ccw=ccw)
        out[ref] = p
    return out


if __name__ == "__main__":
    import sys
    dd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    paths = gen_all(dd)
    for ref, p in paths.items():
        print("wrote %-7s R=%.4f -> %s" % (ref, REFS[ref][0], p))
