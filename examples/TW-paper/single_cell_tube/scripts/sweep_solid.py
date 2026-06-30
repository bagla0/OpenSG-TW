"""WSL/FEniCS: FEniCS-2D-solid Timoshenko 6x6 for the R/h = 1..10 sweep.
Run: conda activate opensg_env_v8 && python -u sweep_solid.py
Order [EA, GA2, GA3, GJ, EI2, EI3]."""
import os
import sys
import time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
os.environ.pop("C1_PENALTY", None)
DATA = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/tube_thesis_314/sweep/data"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

t0 = time.time()
for rh in range(1, 11):
    mesh = os.path.join(DATA, "solid_rh%02d.yaml" % rh)
    sm = SolidBounMesh(mesh)
    mp, _ = sm.material_database
    Cs = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
    Cs = 0.5 * (Cs + Cs.T)
    np.savetxt(os.path.join(DATA, "C6_solid_rh%02d.txt" % rh), Cs,
               header="FEniCS-2D-solid [-45] tube R/h=%d  [EA GA2 GA3 GJ EI2 EI3]" % rh)
    print("rh=%2d  %5.0fs  EA=%.4e GA=%.4e GJ=%.4e EI=%.4e"
          % (rh, time.time() - t0, Cs[0, 0], Cs[1, 1], Cs[3, 3], Cs[4, 4]), flush=True)
print("DONE", flush=True)
