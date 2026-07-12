"""ex3_iea_conv_oml.py -- EXAMPLE 3: IEA-22 r/R=0.2 RM 6-DOF mesh-convergence with the
OML laminate convention (contour on the outer mold line, laminate stacked inward),
consistent with the stored blade yamls and the official OpenSG airfoil convention.

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
from xsec_5v6_master import load_solid, LBL
from oml_ring import load_ring_ref, c6, derr
from taper_common import WINDIO
from opensg_io.converter import load_blade, build_cross_section, emit_opensg_yaml

IB = os.path.abspath(os.path.join(HERE, "..", "iea22_blade", "data"))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
SCR = os.path.join(OUT, "_ex3_scr"); os.makedirs(SCR, exist_ok=True)
So = load_solid(os.path.join(IB, "C6_solid_r020.txt"))
blade = load_blade(WINDIO)
MS = [0.06, 0.045, 0.03, 0.02, 0.015, 0.01]

rows, nn = [], []
for ms in MS:
    cs = build_cross_section(blade, 0.2, mesh_size=ms)
    sp = os.path.join(SCR, "shell_oml_ms%.3f.yaml" % ms)
    emit_opensg_yaml(cs, sp, fraction=0.0)                    # contour ON the OML
    R = load_ring_ref(sp, "oml")                              # laminate stacked inward
    C = c6(R)
    e = derr(C, So)
    rows.append(e); nn.append(len(R["rx"]))
    print("ms=%.3f  nnode=%-4d  diag %%err: %s"
          % (ms, len(R["rx"]), "  ".join("%s%+6.2f" % (LBL[i], e[i]) for i in range(6))), flush=True)

np.savez(os.path.join(OUT, "ex3_iea_conv.npz"),
         mesh_size=np.array(MS), nnode=np.array(nn), diag_err=np.array(rows), labels=LBL)
print("wrote ex3_iea_conv.npz (OML convention)")
