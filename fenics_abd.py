"""
FEniCS OpenSG plate ABD (opensg.core.shell.compute_ABD_matrix) for the same test
laminates as verify_abd.py, to cross-check the JAX MSG-TW compute_ABD_matrix.
Run in WSL (opensg_env_v8).  FEniCS angle is in RADIANS; its 1D SG stacks inward
(-x) from the OML reference (z_ref=0), so compare A and D directly and B up to the
through-thickness sign convention.
"""
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=False, linewidth=140)
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
from opensg.core.shell import compute_ABD_matrix as fenics_abd

ISO = {"iso": {"E": [70e9, 70e9, 70e9], "G": [26.923e9]*3, "nu": [0.3, 0.3, 0.3], "rho": 1800.0}}
COMP = {"m": {"E": [37e9, 9e9, 9e9], "G": [4e9, 4e9, 4e9], "nu": [0.3, 0.3, 0.3], "rho": 1800.0}}
cases = [("isotropic 1-ply", [0.01], [0.0], ["iso"], ISO),
         ("[0/90] unsymmetric", [0.005, 0.005], [0.0, 90.0], ["m", "m"], COMP),
         ("[45/-45] unsymmetric", [0.005, 0.005], [45.0, -45.0], ["m", "m"], COMP)]

for name, thick, angle, names, mat in cases:
    abd = np.asarray(fenics_abd(thick, len(thick), angle, names, mat)[0])
    print(f"\n===== FEniCS OpenSG ABD: {name} =====")
    print("A11,B11,D11 = {:.5e}  {:.5e}  {:.5e}".format(abd[0, 0], abd[0, 3], abd[3, 3]))
    print("B-block:\n", abd[:3, 3:])
