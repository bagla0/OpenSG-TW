"""WSL: compute the FEniCS-solid Timoshenko 6x6 for the thickness factors whose C6_solid_*.txt is
still missing (f=0.8, 1.0).  Solid YAMLs already exist in yaml_solid/.  Sequential (dolfinx)."""
import sys
import os
import time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

STUDY = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/mh104_thickness_study"
RES = STUDY + "/results"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

t0 = time.time()
for f in [0.3, 0.75]:
    t = "f%03d" % int(round(f * 100))
    out = RES + "/C6_solid_%s.txt" % t
    if os.path.exists(out):
        print("[%s] solid exists, skip" % t, flush=True); continue
    sm = SolidBounMesh(STUDY + "/yaml_solid/solid_%s.yaml" % t)
    mp, _ = sm.material_database
    Cs = np.asarray(compute_timo_boun(mp, sm.meshdata)[0]); Cs = 0.5 * (Cs + Cs.T)
    np.savetxt(out, Cs)
    print("[%s] SOLID (%.0fs)  %s" % (t, time.time() - t0, "  ".join("%s=%.4e" % (LBL[i], Cs[i, i]) for i in range(6))), flush=True)
print("done (%.0fs)" % (time.time() - t0), flush=True)
