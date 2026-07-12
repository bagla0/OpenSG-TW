"""ex2_ellipse.py -- EXAMPLE 2: prismatic elliptic 4-CELL tube (3 internal webs), the
multi-cell composite section.  RM 6-DOF ring vs 2-D solid for the isotropic and the
[-45] (m45) laminate, at a THIN and a THICK wall, to show the model holds across the
wall-thickness range on a genuinely multi-cell composite geometry.

  a=1.0, b=0.6;  thin t=0.02 (t/a=0.02),  thick t=0.12 (t/a=0.12)
  -> results/ex2_ellipse.npz  ({mat}_{thin,thick}_{solid,ring}, toa_thin, toa_thick)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELL = os.path.join(HERE, "ellipse")
sys.path.insert(0, HERE); sys.path.insert(0, ELL)
import ell4cell as E

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
NC = 144
CASES = [("thin", 0.02, 6), ("thick", 0.12, 8)]     # (label, wall t, NR through-thickness)


def diagerr(C, So):
    return np.array([100.0 * (C[i, i] - So[i, i]) / So[i, i] for i in range(6)])


store = {}
for lab, T, nr in CASES:
    E.T = float(T); E.NR = int(nr)
    E.gen(NC)
    for mkey in ("iso", "m45"):
        solid, ring = E.homogenize(mkey)
        store["%s_%s_solid" % (mkey, lab)] = solid
        store["%s_%s_ring" % (mkey, lab)] = ring
        e = diagerr(ring, solid)
        print("ellipse %-3s %-5s (t/a=%.3f) diag %%err: %s"
              % (mkey, lab, T / E.A, "  ".join("%s%+6.2f" % (LBL[i], e[i]) for i in range(6))), flush=True)

store["toa_thin"] = np.array(CASES[0][1] / E.A)
store["toa_thick"] = np.array(CASES[1][1] / E.A)
np.savez(os.path.join(OUT, "ex2_ellipse.npz"), **store)
print("wrote ex2_ellipse.npz")
