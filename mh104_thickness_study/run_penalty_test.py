"""WSL: calibrate the generalized C1 penalty on mh104 f=0.2 (the case whose flapwise GA3/EI2 collapse
with the old fixed 1.2e13 window).  Report max_D (the D-block scale) and GA3/EI2 vs solid for the
generalized penalty (BETA*max_D) and a few override values."""
import os
import sys
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import ShellBounMesh

STUDY = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/mh104_thickness_study"
shy = STUDY + "/debug/shell_ref_f020_connect.yaml"
so = np.loadtxt(STUDY + "/results/C6_solid_f020.txt")

shm0 = ShellBounMesh(shy); ABD0, _ = shm0.compute_ABD(frac=0.0)
mD = float(np.max(np.abs(np.asarray(ABD0)[..., 3:, 3:])))
print("mh104 f=0.2:  max_D(D-block)=%.3e   generalized BETA*max_D (BETA=1e7)=%.3e" % (mD, 1e7 * mD), flush=True)
print("solid:  GA3=%.3e  EI2=%.3e  GA2=%.3e\n" % (so[2, 2], so[4, 4], so[1, 1]), flush=True)

for pen in ("gen", "1.2e13", "1e12", "3e11", "1e11"):
    if pen == "gen":
        os.environ.pop("C1_PENALTY", None)
    else:
        os.environ["C1_PENALTY"] = pen
    shm = ShellBounMesh(shy); ABD, _ = shm.compute_ABD(frac=0.0)
    C = np.asarray(shm.compute_timo(ABD)[1]); C = 0.5 * (C + C.T)
    print("penalty=%-8s  GA3=%.3e (%+5.0f%%)  EI2=%.3e (%+5.0f%%)  GA2=%.3e  GJ=%.3e  EA=%.3e" % (
        pen, C[2, 2], 100 * (C[2, 2] - so[2, 2]) / so[2, 2], C[4, 4], 100 * (C[4, 4] - so[4, 4]) / so[4, 4],
        C[1, 1], C[3, 3], C[0, 0]), flush=True)
print("done", flush=True)
