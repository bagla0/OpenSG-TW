"""Recompute ONLY C6_jax_rm_rhXX.txt with mitc_both for R/h=1..10, using the
existing shell meshes.  Leaves C6_jax_kirch (KL) and C6_solid untouched so the
convergence plot updates only the RM curve."""
import os, sys, time
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "tube_45_45", "scripts"))
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
import tube_lib as T

DATA = os.path.join(CC, "tests", "research", "tube_thesis_314", "sweep", "data")
R = 0.0715
N = 3200
t0 = time.time()
for rh in range(1, 11):
    h = R / rh
    mesh = os.path.join(DATA, "shell_rh%02d.yaml" % rh)        # existing N=3200 mesh
    RM, _ = T.homog(mesh, R, h / 2.0, k22_mode="exact", shear="mitc_both")
    RM = 0.5 * (RM + RM.T)
    np.savetxt(os.path.join(DATA, "C6_jax_rm_rh%02d.txt" % rh), RM,
               header="JAX-RM mitc_both [-45] tube R/h=%d center k22=-1/R N=%d  [EA GA2 GA3 GJ EI2 EI3]" % (rh, N))
    print("rh=%2d  %5.0fs  RM GA2=%.4e  GA3=%.4e" % (rh, time.time() - t0, RM[1, 1], RM[2, 2]), flush=True)
print("DONE", flush=True)
