"""Test the JAX Reissner-Mindlin homogenization (strip_RM.rm_timoshenko_6x6) on the mh104 closed
multi-cell airfoil at OML, vs the JAX-Kirchhoff and FEniCS solid (f=0.2)."""
import os
import sys
import traceback
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, CC)
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)

lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
RES = os.path.join(CC, "mh104_thickness_study", "results")
shy = os.path.join(CC, "mh104_thickness_study", "debug", "shell_ref_f020_connect.yaml")
so = np.loadtxt(os.path.join(RES, "C6_solid_f020.txt"))
kf = np.loadtxt(os.path.join(RES, "C6_shell_jax_OML_f020.txt"))

try:
    from strip_RM import rm_timoshenko_6x6
    out = rm_timoshenko_6x6(shy, 0.0)
    C = np.asarray(out[0] if isinstance(out, tuple) else out)
    C = 0.5 * (C + C.T)
    print("RM ran OK.  diagonal (RM | Kirchhoff | solid ; %% vs solid):")
    for i in range(6):
        print("  %-4s RM=%.3e Kirch=%.3e solid=%.3e   RM=%+6.1f%%  Kirch=%+6.1f%%" % (
            lab[i], C[i, i], kf[i, i], so[i, i],
            100 * (C[i, i] - so[i, i]) / abs(so[i, i]), 100 * (kf[i, i] - so[i, i]) / abs(so[i, i])))
except Exception:
    traceback.print_exc()
