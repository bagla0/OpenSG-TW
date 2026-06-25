"""Generate the CCW shell mesh and JAX Kirchhoff + RM OML 6x6 for the new factors f=0.3, 0.75."""
import os
import subprocess
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
sys.path.insert(0, CC)
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml
from strip_RM import rm_timoshenko_6x6

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CC, "mh104_thickness_study", "results")
PY = sys.executable
for fi in (30, 75):
    f = fi / 100.0
    subprocess.run([PY, os.path.join(HERE, "build_ref_yaml.py"), "connect", "f=%.2f" % f], cwd=HERE, capture_output=True)
    shy = os.path.join(HERE, "shell_ref_f%03d_connect.yaml" % fi)
    Ck = np.asarray(solve_tw_from_yaml(shy, frac=0.0)["Timo"]); Ck = 0.5 * (Ck + Ck.T)
    np.savetxt(os.path.join(RES, "C6_shell_jax_OML_f%03d.txt" % fi), Ck, header="JAX-Kirchhoff OML f=%.2f" % f)
    out = rm_timoshenko_6x6(shy, 0.0); Cr = np.asarray(out[0] if isinstance(out, tuple) else out); Cr = 0.5 * (Cr + Cr.T)
    np.savetxt(os.path.join(RES, "C6_shell_rm_OML_f%03d.txt" % fi), Cr, header="JAX-RM OML f=%.2f" % f)
    print("f=%.2f  Kirch EA=%.3e GA3=%.3e   RM GA3=%.3e" % (f, Ck[0, 0], Ck[2, 2], Cr[2, 2]), flush=True)
print("done", flush=True)
