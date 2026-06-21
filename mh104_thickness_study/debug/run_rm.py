"""Run the JAX Reissner-Mindlin homogenization (strip_RM) at OML for every thickness factor and save
results/C6_shell_rm_OML_f0NN.txt, to compare alongside JAX-Kirchhoff and the FEniCS solid."""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, CC)
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from strip_RM import rm_timoshenko_6x6

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CC, "mh104_thickness_study", "results")
for fi in (10, 20, 40, 60, 80, 100):
    shy = os.path.join(HERE, "shell_ref_f%03d_connect.yaml" % fi)
    out = rm_timoshenko_6x6(shy, 0.0)
    C = np.asarray(out[0] if isinstance(out, tuple) else out); C = 0.5 * (C + C.T)
    np.savetxt(os.path.join(RES, "C6_shell_rm_OML_f%03d.txt" % fi), C,
               header="mh104 JAX-Reissner-Mindlin OML  f=%.2f  [EA,GA2,GA3,GJ,EI2,EI3]" % (fi / 100))
    print("f=%.2f RM done" % (fi / 100), flush=True)
print("RM OML sweep complete", flush=True)
