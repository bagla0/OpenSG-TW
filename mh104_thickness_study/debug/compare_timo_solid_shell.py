"""mh104 f=0.2: compare the JAX-Kirchhoff shell Timoshenko 6x6 (CCW-corrected vs old CW) to the
VABS-validated solid (results/C6_solid_f020.txt).  Both referenced at the quarter-chord (mesh origin),
so off-diagonal coupling terms are directly comparable.  The e2 (CW->CCW) fix should flip the sign of
the off-axis-sensitive coupling terms to MATCH the solid.  k22=0 (flat-facet) is now in effect."""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
S = np.loadtxt(os.path.join(CC, "mh104_thickness_study", "results", "C6_solid_f020.txt"))
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0


def run(y):
    C = np.asarray(solve_tw_from_yaml(y, frac=FRAC)["Timo"]); return 0.5 * (C + C.T)


Cc = run(os.path.join(HERE, "shell_ref_f020_connect.yaml"))       # CCW (new default)
Cw = run(os.path.join(HERE, "shell_ref_f020_connect_cw.yaml"))    # CW (old)

print("=== mh104 f=0.2  Kirchhoff vs solid  (frac=%.2f, k22=0) ===" % FRAC)
print("DIAGONAL:")
print("  term  solid         CCW           d%%       CW            d%%")
for i in range(6):
    print("  %-4s %.4e  %.4e %+6.1f%%  %.4e %+6.1f%%" % (
        lab[i], S[i, i], Cc[i, i], 100 * (Cc[i, i] - S[i, i]) / abs(S[i, i]),
        Cw[i, i], 100 * (Cw[i, i] - S[i, i]) / abs(S[i, i])))

print("\nKEY OFF-DIAGONAL COUPLING (sign flips with off-axis ply mirror-imaging):")
print("  i,j  term       solid         CCW           CW          CCW sign vs solid")
pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 3), (3, 4), (3, 5), (4, 5)]
for (i, j) in pairs:
    ok = "OK" if np.sign(Cc[i, j]) == np.sign(S[i, j]) else "FLIP"
    cw = "ok" if np.sign(Cw[i, j]) == np.sign(S[i, j]) else "flip"
    print("  %d,%d  %-8s %+.3e  %+.3e  %+.3e   %s (CW %s)" % (i, j, lab[i] + "-" + lab[j], S[i, j], Cc[i, j], Cw[i, j], ok, cw))

# overall off-diagonal sign agreement
od = [(i, j) for i in range(6) for j in range(6) if i < j]
ccw_ok = sum(np.sign(Cc[i, j]) == np.sign(S[i, j]) for i, j in od)
cw_ok = sum(np.sign(Cw[i, j]) == np.sign(S[i, j]) for i, j in od)
print("\noff-diagonal sign agreement with solid (15 terms):  CCW=%d/15   CW=%d/15" % (ccw_ok, cw_ok))
