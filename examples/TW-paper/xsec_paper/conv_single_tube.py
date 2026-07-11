"""conv_single_tube.py -- mesh-refinement CONVERGENCE of the cross-section 6x6 for the
single-cell [-45] tube: regenerate the SAME geometry (read from shell_center.yaml) at
increasing circumferential element counts N, homogenize with BOTH formulations
(6-DOF Lagrange g23, 5-DOF MITC g23), and report %err vs the fixed 2-D solid
(C6_solid_314.txt).  The solid is N-independent, so the curves show each formulation's
DISCRETIZATION convergence toward its RM cross-section model value.

    python conv_single_tube.py
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LIB = os.path.abspath(os.path.join(HERE, "..", "lib"))
for q in (MITC, REPO, LIB):
    sys.path.insert(0, q)

from gen_meshes import gen_tube_yaml
from xsec_5v6_master import load_ring, load_solid, ring_6dof, ring_5dof, LBL, _row

SC = os.path.abspath(os.path.join(HERE, "..", "single_cell_tube", "data"))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
NS = [40, 80, 160, 320, 640, 1280, 2560]

# read the reference geometry (center-referenced contour) from the bundled N=200 mesh
d = yaml.safe_load(open(os.path.join(SC, "shell_center.yaml")))
n0 = _row(d["nodes"][0])
R_ref = float(np.hypot(n0[0], n0[1]))
layup = [(float(a), float(t)) for (_m, t, a) in d["sections"][0]["layup"]]
mat = d["materials"][0].get("elastic", d["materials"][0])
So = load_solid(os.path.join(SC, "C6_solid_314.txt"))
print("single tube: R_ref=%.5f m, layup=%s" % (R_ref, layup))

rows = []
scratch = os.path.join(OUT, "_conv_meshes"); os.makedirs(scratch, exist_ok=True)
for N in NS:
    p = os.path.join(scratch, "tube_n%d.yaml" % N)
    gen_tube_yaml(p, R_ref, layup=layup, mat=mat, n=N, ccw=True)
    R = load_ring(p)
    C6 = ring_6dof(R); C5 = ring_5dof(R)
    e6 = [100.0 * (C6[i, i] - So[i, i]) / So[i, i] for i in range(6)]
    e5 = [100.0 * (C5[i, i] - So[i, i]) / So[i, i] for i in range(6)]
    rows.append((N, e6, e5))
    print("N=%-5d  6-DOF %s | 5-DOF %s"
          % (N, " ".join("%+5.2f" % v for v in e6), " ".join("%+5.2f" % v for v in e5)))

np.savez(os.path.join(OUT, "conv_single_tube.npz"),
         N=np.array([r[0] for r in rows]),
         err6=np.array([r[1] for r in rows]), err5=np.array([r[2] for r in rows]),
         labels=LBL)
print("\nheader: N  " + "  ".join(LBL))
print("wrote", os.path.join(OUT, "conv_single_tube.npz"))
