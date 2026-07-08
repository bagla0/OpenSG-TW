"""run_ring_prevabs.py -- RM-shell boundary ring 6x6 on the PreVABS webbed-ellipse
contour, both shear schemes, printed next to the centroid-aligned solid 6x6 diagonal.

    python run_ring_prevabs.py <shell_yaml> [solid_S6.npy]
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_ringboun import ring_from_yaml

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
shell_yaml = sys.argv[1]
npz = os.path.join(os.path.dirname(shell_yaml) or ".", "_ring_tmp.npz")

np.set_printoptions(suppress=True, linewidth=160)
for sch, nm in (("full", "full-integ"), ("mitc4_g23", "g23-tied")):
    C = ring_from_yaml(shell_yaml, npz, sch)
    print("\n=== RM-shell ring (%s) 6x6 (x1e9) ===" % nm)
    print(np.round(C / 1e9, 4))
    print("diag:", "  ".join("%s=%.4g" % (LBL[i], C[i, i] / 1e9) for i in range(6)))

if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    S = np.load(sys.argv[2])
    print("\nsolid diag:", "  ".join("%s=%.4g" % (LBL[i], S[i, i] / 1e9) for i in range(6)))
