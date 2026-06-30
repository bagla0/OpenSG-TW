"""FEniCS-2D-solid Timoshenko 6x6 for the Section 3.1.4 single-ply [-45] tube.
Run under WSL:  conda activate opensg_env_v8 && python -u run_solid_314.py
Order [EA, GA2, GA3, GJ, EI2, EI3]."""
import os
import sys
import time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
os.environ.pop("C1_PENALTY", None)

DATA = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/tube_thesis_314/data"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

t0 = time.time()
sm = SolidBounMesh(os.path.join(DATA, "solid_m45.yaml"))
mp, _ = sm.material_database
Cs = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
Cs = 0.5 * (Cs + Cs.T)
np.savetxt(os.path.join(DATA, "C6_solid_314.txt"), Cs,
           header="FEniCS-2D-solid [-45] tube R=7.15cm t=8.682mm  order [EA GA2 GA3 GJ EI2 EI3]")
print("[solid %5.0fs] %s" % (time.time() - t0,
      "  ".join("%s=%.4e" % (LBL[i], Cs[i, i]) for i in range(6))), flush=True)
print("DONE", flush=True)
