"""Investigate C23 (shear-y <-> shear-z coupling): raw value (solid/RM/KL), normalized coupling
C23/sqrt(C22*C33), and %-error across stations. Is the r=0.6 'huge error' real or a small-denom artifact?"""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

VAL = os.path.join(CC, "windio_converter", "validation")
STATIONS = [(round(0.1 * k, 2), "r%03d" % (10 * k)) for k in range(1, 10)] + [(0.95, "r095")]


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


print("C23 = shear-y <-> shear-z coupling  [entry (2,3)]")
print("  r    | solid C23     RM C23       KL C23     | norm(solid)=C23/sqrt(C22*C33) | RM%err   KL%err")
for r, tag in STATIONS:
    sp = os.path.join(VAL, "C6_solid_iea22_%s.txt" % tag)
    sh = os.path.join(VAL, "shell_iea22_%s.yaml" % tag)
    if not (os.path.exists(sp) and os.path.exists(sh)):
        continue
    S = sym(np.loadtxt(sp)); RM = sym(rm_timoshenko_6x6(sh, 0.0, orient=False))
    KL = sym(gradient_junction_kirchhoff(sh, frac=0.0, orient=False)[0])
    sc = S[1, 2]; rc = RM[1, 2]; kc = KL[1, 2]
    norm = sc / np.sqrt(abs(S[1, 1] * S[2, 2]))
    flag = "  <-- r=0.6" if abs(r - 0.6) < 1e-9 else ""
    print("  %.2f | %+.4e  %+.4e  %+.4e | %+.4f                       | %+8.1f %+8.1f%s"
          % (r, sc, rc, kc, norm, 100 * (rc - sc) / sc, 100 * (kc - sc) / sc, flag))

print("\nFor reference, the diagonals it couples (r=0.6): solid C22(shear-y), C33(shear-z):")
S = sym(np.loadtxt(os.path.join(VAL, "C6_solid_iea22_r060.txt")))
print("  C22=%.4e  C33=%.4e  ;  a 'normal' coupling would be O(0.1)*sqrt(C22*C33)=%.3e"
      % (S[1, 1], S[2, 2], 0.1 * np.sqrt(S[1, 1] * S[2, 2])))
