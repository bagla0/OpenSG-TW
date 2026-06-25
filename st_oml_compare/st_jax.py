"""JAX-Kirchhoff (C1-Hermite) + JAX-RM OML 6x6 for st15 and st12, saved to st_oml_compare/data/."""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax")); sys.path.insert(0, CC)
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml
from strip_RM import rm_timoshenko_6x6

DATA = os.path.join(CC, "training data", "opensg-FEniCS", "data")
OUT = os.path.join(CC, "st_oml_compare", "data"); os.makedirs(OUT, exist_ok=True)
for st in ("15", "12"):
    shy = os.path.join(DATA, "1Dshell_%s.yaml" % st)
    Ck = np.asarray(solve_tw_from_yaml(shy, frac=0.0)["Timo"]); Ck = 0.5 * (Ck + Ck.T)
    np.savetxt(os.path.join(OUT, "C6_st%s_jax_kirch.txt" % st), Ck, header="st%s JAX-Kirchhoff OML" % st)
    out = rm_timoshenko_6x6(shy, 0.0); Cr = np.asarray(out[0] if isinstance(out, tuple) else out); Cr = 0.5 * (Cr + Cr.T)
    np.savetxt(os.path.join(OUT, "C6_st%s_jax_rm.txt" % st), Cr, header="st%s JAX-RM OML" % st)
    print("st%s  JAX-Kirch EA=%.3e GA3=%.3e  RM GA3=%.3e" % (st, Ck[0, 0], Ck[2, 2], Cr[2, 2]), flush=True)
print("done", flush=True)
