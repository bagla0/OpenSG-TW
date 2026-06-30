"""Two-cell tube: KL vs RM(mitc) vs RM(mitc_both) diagonal %-error vs 2D-solid.
Shows exactly what mitc_both changes for the paper's multicell tables."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("opensg_jax", ""):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff

DATA = os.path.join(CC, "tests", "research", "multicell_tube", "data")
LBL = ["C11(EA) ", "C22(GA2)", "C33(GA3)", "C44(GJ) ", "C55(EI2)", "C66(EI3)"]


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def pe(m, s):
    return 100.0 * (m - s) / s if s != 0 else float("nan")


CASES = [
    ("ISO  thin  R/h=12.5", "tube2cell_thin.yaml",       "C6_solid_tube2cell_thin.txt",        0.004, False),
    ("ISO  thick R/h=3.1 ", "tube2cell_thick.yaml",      "C6_solid_tube2cell_thick.txt",       0.016, False),
    ("ANI  thin  R/h=12.5", "tube2cell_aniso_thin.yaml",  "C6_solid_tube2cell_aniso_thin.txt",  0.004, True),
    ("ANI  thick R/h=3.1 ", "tube2cell_aniso_thick.yaml", "C6_solid_tube2cell_aniso_thick.txt", 0.016, True),
]

for name, ym, sf, t, aniso in CASES:
    mesh = os.path.join(DATA, ym); S = sym(np.loadtxt(os.path.join(DATA, sf)))
    KL = sym(gradient_junction_kirchhoff(mesh, frac=0.0, dshift=t / 2.0, orient=False)[0])
    Rm = sym(rm_timoshenko_6x6(mesh, 0.0, dshift=t / 2.0, curved=True,
                               shear="mitc", v1shear="mitc", orient=False))
    Rb = sym(rm_timoshenko_6x6(mesh, 0.0, dshift=t / 2.0, curved=True,
                               shear="mitc_both", v1shear="mitc_both", orient=False))
    print("\n=== %s ===" % name)
    print("  term       solid          KL%%       RM(mitc)%%   RM(mitc_both)%%")
    for i in range(6):
        print("  %s %12.4e  %+8.2f   %+9.2f   %+9.2f"
              % (LBL[i], S[i, i], pe(KL[i, i], S[i, i]), pe(Rm[i, i], S[i, i]), pe(Rb[i, i], S[i, i])))
    if aniso:
        print("  C14(e-t)  %12.4e  %+8.2f   %+9.2f   %+9.2f"
              % (S[0, 3], pe(KL[0, 3], S[0, 3]), pe(Rm[0, 3], S[0, 3]), pe(Rb[0, 3], S[0, 3])))
print("\nDONE", flush=True)
