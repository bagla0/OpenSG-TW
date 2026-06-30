"""JAX KL + RM Timoshenko 6x6 for the R/h = 1..10 sweep.
Center reference, k22 = -1/R (exact), refined shell N=3200, single ply [-45].
Order [EA, GA2, GA3, GJ, EI2, EI3]."""
import os
import sys
import time
import numpy as np

TUBE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_45_45\scripts"
sys.path.insert(0, TUBE)
import tube_lib as T
from gen_meshes import gen_tube_yaml, ANI

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tests\research\tube_thesis_314\sweep\data"
R = 0.0715
N = 3200
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

t0 = time.time()
for rh in range(1, 11):
    h = R / rh
    mesh = os.path.join(DATA, "shell_rh%02d.yaml" % rh)
    gen_tube_yaml(mesh, R, layup=[(-45.0, h)], mat=ANI, n=N, ccw=True)
    RM, KF = T.homog(mesh, R, h / 2.0, k22_mode="exact")   # center ref, k22=-1/R
    RM = 0.5 * (RM + RM.T)
    KF = 0.5 * (KF + KF.T)
    np.savetxt(os.path.join(DATA, "C6_jax_kirch_rh%02d.txt" % rh), KF,
               header="JAX-KL [-45] tube R/h=%d center k22=-1/R N=%d  [EA GA2 GA3 GJ EI2 EI3]" % (rh, N))
    np.savetxt(os.path.join(DATA, "C6_jax_rm_rh%02d.txt" % rh), RM,
               header="JAX-RM [-45] tube R/h=%d center k22=-1/R N=%d  [EA GA2 GA3 GJ EI2 EI3]" % (rh, N))
    print("rh=%2d  %5.0fs  KL GA=%.4e  RM GA=%.4e" % (rh, time.time() - t0, KF[1, 1], RM[1, 1]), flush=True)
print("DONE", flush=True)
