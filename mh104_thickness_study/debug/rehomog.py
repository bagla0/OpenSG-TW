"""Re-homogenize every shell case (JAX-Kirchhoff, CCW, k22=0, fixed-IML offset) at OML/center/IML
and overwrite results/C6_shell_jax_*.txt.  Meshes already exist; no regeneration/figures (fast)."""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CC, "mh104_thickness_study", "results")
for fi in (10, 20, 40, 60, 80, 100):
    shy = os.path.join(HERE, "shell_ref_f%03d_connect.yaml" % fi)
    for frac, ref in ((0.0, "OML"), (0.5, "center"), (1.0, "IML")):
        C = np.asarray(solve_tw_from_yaml(shy, frac=frac)["Timo"]); C = 0.5 * (C + C.T)
        np.savetxt(os.path.join(RES, "C6_shell_jax_%s_f%03d.txt" % (ref, fi)), C,
                   header="mh104 JAX-Kirchhoff CCW k22=0 fixed-IML  f=%.2f ref=%s  [EA,GA2,GA3,GJ,EI2,EI3]" % (fi / 100, ref))
    print("f=%.2f done" % (fi / 100), flush=True)
print("re-homogenized all 18 cases", flush=True)
