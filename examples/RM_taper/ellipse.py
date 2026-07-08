"""RM-taper reproducible example -- webbed ELLIPSE tube ([-45] ply, m45).

Multi-cell tapered ellipse with 3 internal shear webs (the paper's most demanding case).
Runs the 6-DOF Reissner-Mindlin shell for the boundary ring AND the tapered segment at both
wall thicknesses (thin t/R=0.02, thick t/R=0.20) and prints the Timoshenko 6x6 vs the
conforming 3-D FEniCS hex-solid reference.

    python examples/RM_taper/ellipse.py

Meshes:  examples/data/rm_taper_ellipse/meshes/shell_<regime>_m45.yaml
Solid ref: examples/data/benchmark/ellipse_solid_m45.npz  (keys <tag>_{L,seg})
"""
import os
import numpy as np
import _rm_common as rm

MESH = os.path.join(rm.CC, "examples", "data", "rm_taper_ellipse", "meshes")
REF = np.load(os.path.join(rm.CC, "examples", "data", "benchmark", "ellipse_solid_m45.npz"))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "ellipse")

cases = []
for regime, tR in [("thin", 0.02), ("thick", 0.20)]:
    tg = "%s_m45" % regime
    cases.append((regime, tR, tg, REF[tg + "_L"], REF[tg + "_seg"]))

rm.run_geometry("Webbed ellipse (multi-cell)", MESH, RES, cases)
