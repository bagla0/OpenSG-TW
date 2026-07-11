"""tube_thick_sweep.py -- THIN-to-THICK wall sweep on the single-cell [-45] tube, the
clean thickness demonstration with a TRUSTED FEniCS 2-D solid reference at every wall
thickness (single_cell_tube/sweep/data: shell_rhNN.yaml + C6_solid_rhNN.txt, R/h=2..10).

Runs the 6-DOF constrained RM ring at each wall and reports the diagonal %err vs the
solid, showing the RM cross-section error is bounded and non-locking from thick
(R/h=2, t/R=0.5) to thin (R/h=10, t/R=0.1) walls.

    python tube_thick_sweep.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for q in (MITC, REPO):
    sys.path.insert(0, q)

from xsec_5v6_master import load_ring, load_solid, ring_6dof, LBL

SW = os.path.abspath(os.path.join(HERE, "..", "single_cell_tube", "sweep", "data"))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
RH = list(range(2, 11))                 # R/h = 2 (thick) .. 10 (thin); t/R = 1/(R/h)

rows = []
for rh in RH:
    shell = os.path.join(SW, "shell_rh%02d.yaml" % rh)
    solid = os.path.join(SW, "C6_solid_rh%02d.txt" % rh)
    if not (os.path.exists(shell) and os.path.exists(solid)):
        print("skip R/h=%d (missing)" % rh); continue
    R = load_ring(shell); So = load_solid(solid); C6 = ring_6dof(R)
    e6 = [100.0 * (C6[i, i] - So[i, i]) / So[i, i] for i in range(6)]
    rows.append((rh, 1.0 / rh, e6))
    print("R/h=%2d (t/R=%.2f)  6-DOF %s" % (rh, 1.0 / rh, " ".join("%+6.2f" % v for v in e6)), flush=True)

np.savez(os.path.join(OUT, "tube_thick_sweep.npz"),
         rh=np.array([r[0] for r in rows]), tR=np.array([r[1] for r in rows]),
         err6=np.array([r[2] for r in rows]), labels=LBL)
print("\nheader: R/h  t/R  " + "  ".join(LBL))
print("wrote", os.path.join(OUT, "tube_thick_sweep.npz"))
