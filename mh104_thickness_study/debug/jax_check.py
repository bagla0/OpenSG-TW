"""Run the JAX MSG-TW Kirchhoff (Hermite C1) and RM shell solvers on the mh104 f=0.2 1D mesh,
OML reference (frac=0), and compare EA to solid 5.50e8 / VABS 5.48e8 / FEniCS-shell 5.59e8."""
import os, sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
sys.path.insert(0, os.path.join(CC, "rm"))
sys.path.insert(0, CC)
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml

YAML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "shell_ref_f020.yaml")
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def show(tag, C):
    print("\n=== %s  (frac=0 OML) ===" % tag)
    print("  diag: " + "  ".join("%s=%.4e" % (LBL[i], C[i, i]) for i in range(6)))


print("YAML:", YAML)
try:
    Ck = np.asarray(solve_tw_from_yaml(YAML, frac=0.0)["Timo"])
    Ck = 0.5 * (Ck + Ck.T)
    show("JAX Kirchhoff (Hermite C1)", Ck)
    np.savetxt(os.path.join(os.path.dirname(__file__), "C6_jax_kirchhoff_f020.txt"), Ck)
except Exception as e:
    import traceback; traceback.print_exc()
    print("Kirchhoff FAILED:", repr(e)[:200])

try:
    from strip_RM import rm_timoshenko_6x6
    Cr = np.asarray(rm_timoshenko_6x6(YAML, frac=0.0))
    Cr = 0.5 * (Cr + Cr.T)
    show("JAX RM (C0, flat k22=0)", Cr)
    np.savetxt(os.path.join(os.path.dirname(__file__), "C6_jax_rm_f020.txt"), Cr)
except Exception as e:
    import traceback; traceback.print_exc()
    print("RM FAILED:", repr(e)[:200])
