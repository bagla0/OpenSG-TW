"""2-cell circular tube (curved-wall multi-cell isotropic): mid-wall circle of
radius R + a diametral vertical web at x=0, splitting it into 2 cells.  Shell mesh
placed DIRECTLY on the mid-wall (no node shifting) -- the centric reference is then
just dshift=t/2 on the ABD (moves no nodes).  Isotropic (e3-sign irrelevant).

Env: TWALL = wall thickness (thin/thick),  TAG = output suffix.  Usage: build_tube2cell.py
"""
import os
import numpy as np
import yaml as _yaml

R = 0.05                                   # mid-wall radius
T = float(os.environ.get("TWALL", "0.004"))   # wall thickness
TAG = os.environ.get("TAG", "")
N = 200                                     # contour segments (mult of 4 -> nodes at +/-90 deg)
NW = 40                                     # web segments
MAT = os.environ.get("MAT", "iso")          # 'iso' or 'aniso' (ud_frp)
ANGLE = float(os.environ.get("ANGLE", "0.0"))   # lamina fiber angle (deg)
if MAT == "aniso":
    matprops = {"E": [37.0e9, 9.0e9, 9.0e9], "G": [4.0e9, 4.0e9, 4.0e9], "nu": [0.28, 0.28, 0.28]}
    rho = 1860.0
else:
    E, NU = 68.9e9, 0.33
    matprops = {"E": [E, E, E], "G": [E / (2.0 * (1.0 + NU))] * 3, "nu": [NU, NU, NU]}
    rho = 2700.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)


class FlowList(list):
    pass


_yaml.add_representer(FlowList, lambda d, x: d.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))

ang = np.array([2.0 * np.pi * k / N for k in range(N)])
pts = [(R * np.cos(a), R * np.sin(a)) for a in ang]
it = N // 4                                # node at (0, +R)
ib = 3 * N // 4                            # node at (0, -R)
nodes = list(pts)
elems = [(k, (k + 1) % N) for k in range(N)]      # closed contour
# diametral web from bottom node to top node
pb, pt = np.array(nodes[ib]), np.array(nodes[it])
prev = ib
for j in range(1, NW + 1):
    cur = it if j == NW else len(nodes)
    if j != NW:
        nodes.append(tuple(pb + (pt - pb) * j / NW))
    elems.append((prev, cur))
    prev = cur

nodes = np.array(nodes)
ori = []
for (a, b) in elems:
    t = nodes[b] - nodes[a]
    e2 = t / (np.linalg.norm(t) + 1e-30)
    e3 = np.array([-e2[1], e2[0]])         # left normal; isotropic -> sign irrelevant
    ori.append([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0, float(e3[0]), float(e3[1]), 0.0])

data = {
    "nodes": [FlowList(["%.10f %.10f 0.0" % (x, y)]) for (x, y) in nodes],
    "elements": [FlowList(["%d %d" % (a + 1, b + 1)]) for (a, b) in elems],
    "sets": {"element": [{"name": "wall", "labels": list(range(1, len(elems) + 1))}]},
    "sections": [{"type": "shell", "elementSet": "wall", "layup": [["mat", float(T), float(ANGLE)]]}],
    "materials": [{"name": "mat", "density": rho, "elastic": matprops}],
    "elementOrientations": [FlowList([float(v) for v in o]) for o in ori],
}
path = os.path.join(OUT, "tube2cell%s.yaml" % TAG)
with open(path, "w") as f:
    _yaml.dump(data, f, sort_keys=False, default_flow_style=False)

# analytical single-cell tube Bredt control (full circle, ignoring the web's small effect)
A_enc = np.pi * R**2
GJ_bredt = 4.0 * A_enc**2 * matprops["G"][0] * T / (2.0 * np.pi * R)
print("wrote %s  (%d nodes, %d elems; R=%.3f t=%.4f h/R=%.3f)"
      % (os.path.basename(path), len(nodes), len(elems), R, T, T / R))
print("single-cell Bredt GJ (no web) = %.2f N.m^2" % GJ_bredt)
