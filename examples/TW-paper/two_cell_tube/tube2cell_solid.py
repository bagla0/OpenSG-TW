"""WSL/FEniCS-2D-solid Timoshenko 6x6 for the 2-cell curved tube, thin + thick."""
import os
import sys
import time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
os.environ.pop("C1_PENALTY", None)
DATA = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/multicell_tube/data"
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

t0 = time.time()
for tag in ("thin", "thick"):
    sm = SolidBounMesh(os.path.join(DATA, "solid_tube2cell_%s.yaml" % tag))
    mp, _ = sm.material_database
    Cs = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
    Cs = 0.5 * (Cs + Cs.T)
    np.savetxt(os.path.join(DATA, "C6_solid_tube2cell_%s.txt" % tag), Cs,
               header="FEniCS-2D-solid 2-cell tube %s [EA GA2 GA3 GJ EI2 EI3]" % tag)
    print("%-5s %4.0fs:  EA=%.4e GA2=%.4e GA3=%.4e GJ=%.4e EI2=%.4e EI3=%.4e"
          % (tag, time.time() - t0, Cs[0, 0], Cs[1, 1], Cs[2, 2], Cs[3, 3], Cs[4, 4], Cs[5, 5]), flush=True)
print("DONE", flush=True)
