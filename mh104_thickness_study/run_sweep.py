"""WSL sweep: for each wall-thickness factor, compute the OpenSG FEniCS Timoshenko 6x6 for the
2D SOLID and the 1D SHELL (OML=frac 0 and CENTER=frac 0.5, both on the OML contour). Saves each C6
to results/.  Order = [ext, shear2, shear3, twist, bend2, bend3] (solid + shell agree, = VABS order).
Sequential (one process) -- concurrent dolfinx runs segfault."""
import sys, os, time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import SolidBounMesh, ShellBounMesh
from opensg.core.solid import compute_timo_boun

STUDY = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/mh104_thickness_study"
RES = STUDY + "/results"
FACTORS = [0.2, 0.4, 0.6, 0.8, 1.0]
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def tag(f):
    return "f%03d" % int(round(f * 100))


def diag(C):
    return "  ".join("%s=%.4e" % (LBL[i], C[i, i]) for i in range(6))


t0 = time.time()
for f in FACTORS:
    t = tag(f)
    # ---- 2D solid ----
    sm = SolidBounMesh(STUDY + "/yaml_solid/solid_%s.yaml" % t)
    mp, _ = sm.material_database
    Cs = np.asarray(compute_timo_boun(mp, sm.meshdata)[0]); Cs = 0.5 * (Cs + Cs.T)
    np.savetxt(RES + "/C6_solid_%s.txt" % t, Cs)
    print("[%s] SOLID  (%.0fs)  %s" % (t, time.time() - t0, diag(Cs)), flush=True)
    # ---- 1D shell: OML (frac 0) + center (frac 0.5) ----
    shm = ShellBounMesh(STUDY + "/yaml_shell/shell_%s.yaml" % t)
    for frac, ref in [(0.0, "oml"), (0.5, "center")]:
        ABD, mass = shm.compute_ABD(frac=frac)
        Deff = shm.compute_timo(ABD)[1]
        C = 0.5 * (np.asarray(Deff) + np.asarray(Deff).T)
        np.savetxt(RES + "/C6_shell_%s_%s.txt" % (ref, t), C)
        print("[%s] SHELL-%-6s (%.0fs)  %s" % (t, ref, time.time() - t0, diag(C)), flush=True)
print("\nsweep complete (%.0fs)." % (time.time() - t0), flush=True)
