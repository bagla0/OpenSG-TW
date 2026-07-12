"""ex1_twocell.py -- EXAMPLE 1 for the RM cross-section paper: webbed two-cell tube.
Headline RM 6-DOF ring vs 2-D solid for BOTH the isotropic and the [-45] (m45) laminate,
plus an RM mesh-convergence sweep (contour element count N) on the isotropic section.

  isotropic : R=0.05, t=0.004, E=68.9 GPa, nu=0.33
  m45       : R=0.05, t=0.004, [-45] ud-frp (E=[37,9,9] GPa, G=4 GPa, nu=0.28)

Headline uses the trusted existing shell yamls (consistent with the solid refs); the
convergence sweep regenerates the isotropic two-cell contour at increasing N (e3-sign
irrelevant for iso) and reports the RM 6-DOF diagonal %err vs the 2-D solid.

  -> results/ex1_twocell.npz  (iso_solid,iso_c6, m45_solid,m45_c6)
  -> results/ex1_conv.npz     (N, diag_err[N,6], labels)
"""
import os
import sys

import numpy as np
import yaml as _yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xsec_5v6_master import load_ring, load_solid, ring_6dof, LBL

TC = os.path.abspath(os.path.join(HERE, "..", "two_cell_tube", "data"))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
SCR = os.path.join(OUT, "_ex1_scr"); os.makedirs(SCR, exist_ok=True)
R, T = 0.05, 0.004
ISO = {"E": [68.9e9] * 3, "G": [68.9e9 / (2 * 1.33)] * 3, "nu": [0.33] * 3, "rho": 2700.0}


class FL(list):
    pass


_yaml.add_representer(FL, lambda d, x: d.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))


def gen_2cell_iso(N):
    """isotropic webbed two-cell contour at N circumferential segments -> yaml path."""
    NW = max(8, N // 5)
    ang = [2.0 * np.pi * k / N for k in range(N)]
    nodes = [(R * np.cos(a), R * np.sin(a)) for a in ang]
    elems = [(k, (k + 1) % N) for k in range(N)]
    it, ib = N // 4, 3 * N // 4                       # top (0,+R), bottom (0,-R)
    pb, pt = np.array(nodes[ib]), np.array(nodes[it])
    prev = ib
    for j in range(1, NW + 1):
        cur = it if j == NW else len(nodes)
        if j != NW:
            nodes.append(tuple(pb + (pt - pb) * j / NW))
        elems.append((prev, cur)); prev = cur
    nodes = np.array(nodes)
    ori = []
    for (a, b) in elems:
        t = nodes[b] - nodes[a]; e2 = t / (np.linalg.norm(t) + 1e-30)
        e3 = np.array([-e2[1], e2[0]])
        ori.append([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0, float(e3[0]), float(e3[1]), 0.0])
    data = {
        "nodes": [FL(["%.10f %.10f 0.0" % (x, y)]) for (x, y) in nodes],
        "elements": [FL(["%d %d" % (a + 1, b + 1)]) for (a, b) in elems],
        "sets": {"element": [{"name": "wall", "labels": list(range(1, len(elems) + 1))}]},
        "sections": [{"type": "shell", "elementSet": "wall", "layup": [["mat", float(T), 0.0]]}],
        "materials": [{"name": "mat", "density": ISO["rho"],
                       "elastic": {"E": FL(ISO["E"]), "G": FL(ISO["G"]), "nu": FL(ISO["nu"])}}],
        "elementOrientations": [FL([float(v) for v in o]) for o in ori],
    }
    p = os.path.join(SCR, "iso_N%d.yaml" % N)
    _yaml.dump(data, open(p, "w"), sort_keys=False, default_flow_style=False)
    return p


def diagerr(C, So):
    return np.array([100.0 * (C[i, i] - So[i, i]) / So[i, i] for i in range(6)])


# ---------------- headline 6x6 (existing trusted yamls) ----------------
iso_So = load_solid(os.path.join(TC, "C6_solid_tube2cell_thin.txt"))
iso_C6 = ring_6dof(load_ring(os.path.join(TC, "tube2cell_thin.yaml")))
m45_So = load_solid(os.path.join(TC, "C6_solid_tube2cell_aniso_thin.txt"))
m45_C6 = ring_6dof(load_ring(os.path.join(TC, "tube2cell_aniso_thin.yaml")))
print("### two-cell ISO  diag %err:", "  ".join("%s%+6.2f" % (LBL[i], diagerr(iso_C6, iso_So)[i]) for i in range(6)))
print("### two-cell m45  diag %err:", "  ".join("%s%+6.2f" % (LBL[i], diagerr(m45_C6, m45_So)[i]) for i in range(6)))
np.savez(os.path.join(OUT, "ex1_twocell.npz"), iso_solid=iso_So, iso_c6=iso_C6,
         m45_solid=m45_So, m45_c6=m45_C6)

# ---------------- RM convergence (isotropic) ----------------
Ns = [40, 80, 160, 320, 640, 1280]
rows = []
for N in Ns:
    C6 = ring_6dof(load_ring(gen_2cell_iso(N)))
    e = diagerr(C6, iso_So); rows.append(e)
    print("N=%-5d nel~%-5d  diag %%err: %s" % (N, N + max(8, N // 5),
          "  ".join("%s%+6.2f" % (LBL[i], e[i]) for i in range(6))), flush=True)
np.savez(os.path.join(OUT, "ex1_conv.npz"), N=np.array(Ns), diag_err=np.array(rows), labels=LBL)
print("wrote ex1_twocell.npz + ex1_conv.npz")
