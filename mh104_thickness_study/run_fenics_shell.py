"""WSL: FEniCS MSG-shell (ShellBounMesh) Timoshenko 6x6 at OML (frac=0) on the CCW-corrected meshes
(debug/shell_ref_f0NN_connect.yaml), for f=0.1..0.6.  Saves results/C6_fenics_shell_OML_f0NN.txt."""
import sys
import os
import time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import ShellBounMesh

STUDY = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/mh104_thickness_study"
RES = STUDY + "/results"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

t0 = time.time()
for fi in (10, 20, 40, 60):
    t = "f%03d" % fi
    shy = STUDY + "/debug/shell_ref_%s_connect.yaml" % t
    try:
        shm = ShellBounMesh(shy)
        ABD, _ = shm.compute_ABD(frac=0.0)
        Deff = np.asarray(shm.compute_timo(ABD)[1]); C = 0.5 * (Deff + Deff.T)
        np.savetxt(RES + "/C6_fenics_shell_OML_%s.txt" % t, C,
                   header="mh104 FEniCS-shell OML (CCW mesh)  f=%.2f  [EA,GA2,GA3,GJ,EI2,EI3]" % (fi / 100))
        print("[%s] FEniCS-shell (%.0fs)  %s" % (t, time.time() - t0,
              "  ".join("%s=%.4e" % (LBL[i], C[i, i]) for i in range(6))), flush=True)
    except Exception as e:
        import traceback
        print("[%s] FAILED: %s" % (t, e), flush=True); traceback.print_exc()
print("done (%.0fs)" % (time.time() - t0), flush=True)
