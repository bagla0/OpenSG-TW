"""Single [-45] tube at R/h=2 and R/h=10: RM(mitc_both) vs published solid, plus
old RM(mitc) from file and solid-drift check.  Confirms the fixed pipeline and
gives the new RM column for tab:thick_wall_r2 / tab:thin_wall_r10."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "tube_45_45", "scripts"))
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
import tube_lib as T

DATA = os.path.join(CC, "tests", "research", "tube_thesis_314", "sweep", "data")
R = 0.0715


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def pe(m, s):
    return 100.0 * (m - s) / s if s != 0 else float("nan")


TERMS = [("C11", 0, 0), ("C14", 0, 3), ("C22", 1, 1), ("C25", 1, 4),
         ("C33", 2, 2), ("C36", 2, 5), ("C44", 3, 3), ("C55", 4, 4), ("C66", 5, 5)]
# published solid (paper tables) and paper RM column
PUB = {
    2:  dict(S=[1.9730e8, -4.0737e6, 6.8641e7, 2.1084e6, 6.8641e7, 2.1084e6, 6.7459e5, 5.4042e5, 5.4042e5],
             RM=[-0.39, -5.38, -7.98, -4.92, -7.98, -4.92, 7.16, -4.99, -4.99]),
    10: dict(S=[3.9311e7, -7.7255e5, 1.2036e7, 3.8697e5, 1.2036e7, 3.8697e5, 1.2274e5, 1.0076e5, 1.0076e5],
             RM=[-0.01, -0.21, -0.43, -0.23, -0.43, -0.23, 0.33, -0.20, -0.20]),
}

for rh in (2, 10):
    h = R / rh
    mesh = os.path.join(DATA, "shell_rh%02d.yaml" % rh)
    Rb = sym(T.homog(mesh, R, h / 2.0, k22_mode="exact", shear="mitc_both")[0])
    Rm_old = sym(np.loadtxt(os.path.join(DATA, "C6_jax_rm_rh%02d.txt" % rh)))  # existing = old mitc
    Scur = sym(np.loadtxt(os.path.join(DATA, "C6_solid_rh%02d.txt" % rh)))
    Sp = PUB[rh]["S"]; prm = PUB[rh]["RM"]
    print("\n==================  R/h = %d  ==================" % rh)
    print("  term   S_pub        S_cur(file)  drift%%   paperRM   RMold(mitc,vsPub)  RMnew(both,vsPub)")
    for k, (lab, i, j) in enumerate(TERMS):
        print("  %-4s %11.4e  %11.4e  %+5.2f   %+6.2f    %+7.2f          %+7.2f"
              % (lab, Sp[k], Scur[i, j], pe(Scur[i, j], Sp[k]), prm[k],
                 pe(Rm_old[i, j], Sp[k]), pe(Rb[i, j], Sp[k])))
print("\nDONE", flush=True)
