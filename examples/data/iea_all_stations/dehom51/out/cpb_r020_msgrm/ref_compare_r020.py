"""ref_compare_r020.py -- r=0.2 Timoshenko 6x6 diagonal at the TRUE mid-surface reference
(contour on the wall mid-line, fraction=0.5) vs the OML reference (fraction=0.0), both
windIO-native, both with the MSG-RM 8x8 wall law, against the SAME 2-D solid.  Prints the
diagonal %errors so the paper can state the OML-vs-mid comparison with real numbers.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
sys.path.insert(0, XSEC)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax

jax.config.update("jax_enable_x64", True)
import dehom_rm

MID = os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml")        # fraction 0.5
OML = os.path.join(ROOT, "shell51", "1d_yaml_oml51", "iea_s10_shell.yaml")   # fraction 0.0
SOLID = os.path.join(ROOT, "..", "..", "TW-paper", "iea22_blade", "data", "C6_solid_r020.txt")

So = np.loadtxt(SOLID)
if So.shape != (6, 6):
    So = So.reshape(6, 6)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
dsol = np.diag(So)

for tag, yml in (("mid-surface", MID), ("OML", OML)):
    B = dehom_rm.build_rm_bundle(yml)
    C = np.asarray(B["Timo"])
    d = np.diag(C)
    err = 100.0 * (d - dsol) / dsol
    print("%-12s ref=%-7s  diag %%err vs solid:  %s"
          % (tag, B.get("ref"), "  ".join("%s %+6.2f" % (LBL[i], err[i]) for i in range(6))),
          flush=True)
print("\nsolid diag:", "  ".join("%s %.4g" % (LBL[i], dsol[i]) for i in range(6)))
