"""ex3_iea_conv.py -- EXAMPLE 3: IEA-22 blade cross-section at r/R=0.2, RM 6-DOF ring
mesh-CONVERGENCE study.  The 1-D shell contour is built from the windIO section at a
range of mesh sizes; the RM 6-DOF diagonal %err vs the fixed 2-D solid (PreVABS) settles
to the shell-model value, confirming the reported error is model, not discretization.

  -> results/ex3_iea_conv.npz  (mesh_size, nnode, diag_err[.,6], labels)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TAPER = os.path.abspath(os.path.join(HERE, "..", "..", "taper"))
for q in (MITC, REPO, TAPER, os.path.expanduser("~/OpenSG_io")):
    sys.path.insert(0, q)
from xsec_5v6_master import load_ring, load_solid, ring_6dof, LBL
from taper_common import WINDIO
from opensg_io.converter import load_blade, build_cross_section, emit_opensg_yaml

IB = os.path.abspath(os.path.join(HERE, "..", "iea22_blade", "data"))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
SCR = os.path.join(OUT, "_ex3_scr"); os.makedirs(SCR, exist_ok=True)
R = 0.2
So = load_solid(os.path.join(IB, "C6_solid_r020.txt"))
blade = load_blade(WINDIO)
MS = [0.06, 0.045, 0.03, 0.02, 0.015, 0.01]

rows, nn = [], []
for ms in MS:
    cs = build_cross_section(blade, R, mesh_size=ms)
    sp = os.path.join(SCR, "shell_ms%.3f.yaml" % ms); emit_opensg_yaml(cs, sp)
    ring = load_ring(sp); C6 = ring_6dof(ring)
    e = np.array([100.0 * (C6[i, i] - So[i, i]) / So[i, i] for i in range(6)])
    rows.append(e); nn.append(len(ring["rx"]))
    print("ms=%.3f  nnode=%-4d  diag %%err: %s"
          % (ms, len(ring["rx"]), "  ".join("%s%+6.2f" % (LBL[i], e[i]) for i in range(6))), flush=True)

np.savez(os.path.join(OUT, "ex3_iea_conv.npz"),
         mesh_size=np.array(MS), nnode=np.array(nn), diag_err=np.array(rows), labels=LBL)
print("wrote ex3_iea_conv.npz")
