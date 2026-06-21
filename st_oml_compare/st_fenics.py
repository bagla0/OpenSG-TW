"""WSL: FEniCS 2D-SOLID (VABS-equivalent reference) and FEniCS-SHELL (generalized penalty, OML) 6x6
for st15 and st12.  Saved to st_oml_compare/data/."""
import os
import sys
import time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG); os.chdir(PKG)
os.environ.pop("C1_PENALTY", None)
from opensg.mesh.segment import SolidBounMesh, ShellBounMesh
from opensg.core.solid import compute_timo_boun

DATA = PKG + "/data"
OUT = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/st_oml_compare/data"
solid_yaml = {"15": DATA + "/2Dsolid_VABS_15.yaml", "12": DATA + "/2Dsolid_12.yaml"}
t0 = time.time()
for st in ("15", "12"):
    # 2D solid reference
    try:
        sm = SolidBounMesh(solid_yaml[st]); mp, _ = sm.material_database
        Cs = np.asarray(compute_timo_boun(mp, sm.meshdata)[0]); Cs = 0.5 * (Cs + Cs.T)
        np.savetxt(OUT + "/C6_st%s_solid.txt" % st, Cs, header="st%s FEniCS 2D-solid (VABS ref)" % st)
        print("st%s SOLID (%.0fs)  EA=%.3e GA3=%.3e EI2=%.3e" % (st, time.time() - t0, Cs[0, 0], Cs[2, 2], Cs[4, 4]), flush=True)
    except Exception as e:
        print("st%s solid FAILED: %s" % (st, e), flush=True)
    # FEniCS shell (generalized penalty), OML
    try:
        shm = ShellBounMesh(DATA + "/1Dshell_%s.yaml" % st); ABD, _ = shm.compute_ABD(frac=0.0)
        Cf = np.asarray(shm.compute_timo(ABD)[1]); Cf = 0.5 * (Cf + Cf.T)
        np.savetxt(OUT + "/C6_st%s_fenics_shell.txt" % st, Cf, header="st%s FEniCS-shell OML genpenalty" % st)
        print("st%s FEshell (%.0fs)  EA=%.3e GA3=%.3e EI2=%.3e" % (st, time.time() - t0, Cf[0, 0], Cf[2, 2], Cf[4, 4]), flush=True)
    except Exception as e:
        print("st%s FEshell FAILED: %s" % (st, e), flush=True)
print("done (%.0fs)" % (time.time() - t0), flush=True)
