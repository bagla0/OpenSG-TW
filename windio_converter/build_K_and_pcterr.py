"""Write the Timoshenko stiffness in plain .K convention (bare 6x6, no row/col labels) for 2D-solid, RM, KL,
and the %-error 6x6 (RM-vs-solid, KL-vs-solid) per station -- to .dat files and per-station .K files."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

VAL = os.path.join(CC, "windio_converter", "validation")
KDIR = os.path.join(VAL, "K_matrices"); os.makedirs(KDIR, exist_ok=True)
STATIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def mat6(M):
    return "\n".join(" ".join("% .6e" % M[i, j] for j in range(6)) for i in range(6))


def pct6(M, S):
    # %-error only where the term is non-negligible; |term| < 1e6 -> tiny denominator, % is noise -> 0.
    out = []
    for i in range(6):
        r = []
        for j in range(6):
            r.append("% 9.2f" % (100 * (M[i, j] - S[i, j]) / S[i, j]) if abs(S[i, j]) >= 1.0e6 else "% 9.2f" % 0.0)
        out.append(" ".join(r))
    return "\n".join(out)


rows = []
for r in STATIONS:
    tag = "r%03d" % round(r * 100)
    shell = os.path.join(VAL, "shell_iea22_%s.yaml" % tag)
    RM = sym(rm_timoshenko_6x6(shell, 0.0, orient=False))
    KL = sym(gradient_junction_kirchhoff(shell, frac=0.0, orient=False)[0])
    sp = os.path.join(VAL, "C6_solid_iea22_%s.txt" % tag)
    S = sym(np.loadtxt(sp)) if os.path.exists(sp) else None
    rows.append((r, tag, S, RM, KL))
    # per-station .K files (bare 6x6, VABS convention)
    np.savetxt(os.path.join(KDIR, "iea22_%s_RM.K" % tag), RM, fmt="% .6e")
    np.savetxt(os.path.join(KDIR, "iea22_%s_KL.K" % tag), KL, fmt="% .6e")
    if S is not None:
        np.savetxt(os.path.join(KDIR, "iea22_%s_solid.K" % tag), S, fmt="% .6e")

# stiffness matrices .dat (bare 6x6 blocks)
with open(os.path.join(VAL, "iea22_stiffness_K.dat"), "w") as f:
    f.write("# IEA-22-280-RWT Timoshenko stiffness 6x6 (.K convention, bare matrix). Order: axial, shear-y,\n")
    f.write("# shear-z, twist, bend-y, bend-z. OML reference. SI (N, N*m, N*m^2).\n")
    for (r, tag, S, RM, KL) in rows:
        for name, M in (("2D-solid", S), ("RM", RM), ("KL", KL)):
            f.write("\n# r=%.2f  %s\n" % (r, name))
            f.write((mat6(M) if M is not None else "(solid pending)") + "\n")

# %-error 6x6 .dat (RM and KL)
for name, mi, fn in (("RM", 3, "iea22_pcterr_RM.dat"), ("KL", 4, "iea22_pcterr_KL.dat")):
    with open(os.path.join(VAL, fn), "w") as f:
        f.write("# IEA-22 %s %%-error vs 2D-solid, full 6x6 (structural zeros shown as 0). Order: axial,\n" % name)
        f.write("# shear-y, shear-z, twist, bend-y, bend-z.\n")
        for (r, tag, S, RM, KL) in rows:
            M = (RM, KL)[mi - 3]
            f.write("\n# r=%.2f\n" % r)
            f.write((pct6(M, S) if S is not None else "(solid pending)") + "\n")

# console: diagonal %-error table
LBL = ["C11", "C22", "C33", "C44", "C55", "C66"]
for name, mi in (("RM", 3), ("KL", 4)):
    print("\n##### %s diagonal %%-error vs 2D-solid (axial,shear-y,shear-z,twist,bend-y,bend-z) #####" % name)
    print("  r    " + " ".join("%8s" % L for L in LBL) + "   max|err|")
    for (r, tag, S, RM, KL) in rows:
        if S is None:
            print("  %.2f   (solid pending)" % r); continue
        M = (RM, KL)[mi - 3]
        e = [100 * (M[i, i] - S[i, i]) / S[i, i] for i in range(6)]
        print("  %.2f " % r + " ".join("%+8.2f" % x for x in e) + "   %.1f" % max(abs(x) for x in e))
print("\nwrote iea22_stiffness_K.dat, iea22_pcterr_RM.dat, iea22_pcterr_KL.dat, K_matrices/*.K -> validation/")
