"""Two-cell tube: RM(mitc_both) %-error vs the PUBLISHED solid (keep solid+KL columns).
Prints the new RM column to drop into the paper, alongside the current paper RM column."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("opensg_jax", ""):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6

DATA = os.path.join(CC, "tests", "research", "multicell_tube", "data")


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def pe(m, s):
    return 100.0 * (m - s) / s if s != 0 else float("nan")


LBL = ["C11", "C22", "C33", "C44", "C55", "C66"]
# (name, yaml, t, S_pub[6], S_pub_C14, paperRM[6 or 7])
CASES = [
    ("ISO  thin  R/h=12.5", "tube2cell_thin.yaml", 0.004,
     [1.1303e8, 1.6481e7, 2.5824e7, 8.1830e4, 1.2868e5, 1.0841e5], None,
     ["+1.0", "-1.1", "-0.2", "-0.3", "+2.0", "-0.1"]),
    ("ISO  thick R/h=3.1 ", "tube2cell_thick.yaml", 0.016,
     [4.3834e8, 7.1800e7, 1.0783e8, 3.3883e5, 4.9738e5, 4.4588e5], None,
     ["+4.2", "-6.8", "-3.2", "+0.3", "+6.2", "-1.6"]),
    ("ANI  thin  R/h=12.5", "tube2cell_aniso_thin.yaml", 0.004,
     [2.0012e7, 4.5128e6, 7.3360e6, 2.3558e4, 2.2524e4, 1.9206e4], -2.1143e5,
     ["+1.0", "-2.0", "-0.2", "-0.3", "+1.5", "-0.2", "-0.3"]),
    ("ANI  thick R/h=3.1 ", "tube2cell_aniso_thick.yaml", 0.016,
     [7.7731e7, 2.2423e7, 3.1235e7, 1.0052e5, 8.9584e4, 7.9529e4], -8.7578e5,
     ["+4.0", "-7.6", "-3.3", "-0.2", "+4.3", "-2.2", "-3.6"]),
]

for name, ym, t, Sp, Sp14, prm in CASES:
    mesh = os.path.join(DATA, ym)
    Rb = sym(rm_timoshenko_6x6(mesh, 0.0, dshift=t / 2.0, curved=True,
                               shear="mitc_both", v1shear="mitc_both", orient=False))
    print("\n=== %s ===" % name)
    print("  term   S_pub(VABS)   paperRM%   newRM(mitc_both)%  (vs published solid)")
    for i in range(6):
        print("  %-4s  %11.4e   %6s     %+8.2f" % (LBL[i], Sp[i], prm[i], pe(Rb[i, i], Sp[i])))
    if Sp14 is not None:
        print("  C14   %11.4e   %6s     %+8.2f" % (Sp14, prm[6], pe(Rb[0, 3], Sp14)))
print("\nDONE", flush=True)
