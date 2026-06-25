"""%-error of the REFINED (mesh_size 0.0025) shell RM/KL vs the 2D-solid, all stations, .K-style 6x6 +
diagonal table. Writes iea22_pcterr_refined_RM.dat / _KL.dat."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

VAL = os.path.join(CC, "windio_converter", "validation")
REF = os.path.join(VAL, "refined")
STATIONS = [(round(0.1 * k, 2), "r%03d" % (10 * k)) for k in range(1, 10)] + [(0.95, "r095")]
DIAG = ["C11", "C22", "C33", "C44", "C55", "C66"]


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def pct6(M, S):
    # %-error is meaningful only where the stiffness term is non-negligible; for |term| < 1e6 the
    # denominator is tiny and the % is noise -> report 0.
    out = []
    for i in range(6):
        row = []
        for j in range(6):
            row.append("% 9.2f" % (100 * (M[i, j] - S[i, j]) / S[i, j]) if abs(S[i, j]) >= 1.0e6 else "% 9.2f" % 0.0)
        out.append(" ".join(row))
    return "\n".join(out)


rows = []
for r, tag in STATIONS:
    sh = os.path.join(REF, "shell_iea22_%s_ms0p0025.yaml" % tag)
    sp = os.path.join(VAL, "C6_solid_iea22_%s.txt" % tag)
    if not (os.path.exists(sh) and os.path.exists(sp)):
        print("skip %s" % tag, flush=True); continue
    RM = sym(rm_timoshenko_6x6(sh, 0.0, orient=False))
    KL = sym(gradient_junction_kirchhoff(sh, frac=0.0, orient=False)[0])
    S = sym(np.loadtxt(sp))
    rows.append((r, S, RM, KL))

for name, mi, fn in (("RM", 2, "iea22_pcterr_refined_RM.dat"), ("KL", 3, "iea22_pcterr_refined_KL.dat")):
    with open(os.path.join(VAL, fn), "w") as f:
        f.write("# IEA-22 %s %%-error vs 2D-solid, REFINED 1D shell (mesh_size=0.0025), full 6x6 (zeros=0)\n" % name)
        for (r, S, RM, KL) in rows:
            f.write("\n# r=%.2f\n" % r + pct6((RM, KL)[mi - 2], S) + "\n")
    print("\n##### %s diagonal %%-error vs 2D-solid -- REFINED shell (mesh_size 0.0025) #####" % name, flush=True)
    print("  r    " + " ".join("%8s" % d for d in DIAG) + "   max|err|", flush=True)
    for (r, S, RM, KL) in rows:
        M = (RM, KL)[mi - 2]
        e = [100 * (M[i, i] - S[i, i]) / S[i, i] for i in range(6)]
        print("  %.2f " % r + " ".join("%+8.2f" % x for x in e) + "   %.1f" % max(abs(x) for x in e), flush=True)
print("\nwrote iea22_pcterr_refined_RM.dat, iea22_pcterr_refined_KL.dat -> validation/", flush=True)
