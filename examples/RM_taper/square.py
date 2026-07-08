"""RM-taper reproducible example -- SQUARE tube ([-45] ply, m45).

Runs the 6-DOF Reissner-Mindlin shell for the boundary ring AND the tapered segment at both
wall thicknesses (thin t/R=0.02, thick t/R=0.20, taper ratio 0.7) and prints the Timoshenko
6x6 vs the conforming 3-D FEniCS solid reference.

    python examples/RM_taper/square.py

Meshes:  examples/data/taper_square/meshes/shell_<regime>_m45_aR070.yaml
Solid ref: examples/data/benchmark/taper_square_solid_m45.npz  (keys <tag>_{L,seg})
"""
import os
import numpy as np
import _rm_common as rm

MESH = os.path.join(rm.CC, "examples", "data", "taper_square", "meshes")
REF = np.load(os.path.join(rm.CC, "examples", "data", "benchmark", "taper_square_solid_m45.npz"))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "square")

cases = []
for regime, tR in [("thin", 0.02), ("thick", 0.20)]:
    tg = "%s_m45_aR070" % regime
    cases.append((regime, tR, tg, REF[tg + "_L"], REF[tg + "_seg"]))

rm.run_geometry("Square tube", MESH, RES, cases)
