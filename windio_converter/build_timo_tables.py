"""Emit (1) the full Timoshenko 6x6 (2D-solid, RM, KL) per station -> iea22_timo_full.dat, and
(2) two %-diff tables (RM and KL vs solid) over ALL NONZERO Timo terms -> iea22_pcterr_RM.dat / _KL.dat."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

VAL = os.path.join(CC, "windio_converter", "validation")
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
STATIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def nm(i, j):
    return LBL[i] if i == j else "C%d%d" % (i + 1, j + 1)


rows = []
for r in STATIONS:
    tag = "r%03d" % round(r * 100)
    sp = os.path.join(VAL, "C6_solid_iea22_%s.txt" % tag)
    if not os.path.exists(sp):
        print("skip r=%.2f (solid pending)" % r); continue
    S = sym(np.loadtxt(sp))
    shell = os.path.join(VAL, "shell_iea22_%s.yaml" % tag)
    RM = sym(rm_timoshenko_6x6(shell, 0.0, orient=False))
    KL = sym(gradient_junction_kirchhoff(shell, frac=0.0, orient=False)[0])
    rows.append((r, S, RM, KL))

# ---- (1) full 6x6 matrices ----
with open(os.path.join(VAL, "iea22_timo_full.dat"), "w") as f:
    f.write("# IEA-22-280-RWT Timoshenko 6x6 stiffness  [order: EA GA2 GA3 GJ EI2 EI3]  (OML reference, frac=0)\n")
    f.write("# Per station: the 2D-solid (FEniCS/VABS-equiv), JAX MSG-RM, and JAX MSG-Kirchhoff 6x6 matrices.\n")
    for (r, S, RM, KL) in rows:
        f.write("\n# ================= station r = %.2f =================\n" % r)
        for name, M in (("2D-SOLID", S), ("RM", RM), ("KL", KL)):
            f.write("# %s\n" % name)
            for i in range(6):
                f.write("  " + " ".join("%15.6e" % M[i, j] for j in range(6)) + "\n")

# ---- nonzero-term set (diagonals + couplings with max|normalized| > 0.03) ----
terms = [(i, j) for i in range(6) for j in range(i, 6)]
sig = []
for (i, j) in terms:
    mx = max(abs(S[i, j]) / np.sqrt(abs(S[i, i] * S[j, j])) for (_, S, _, _) in rows)
    if i == j or mx > 0.03:
        sig.append((i, j))


def write_pct(fname, mi, title):
    out = []
    out.append("# IEA-22 %s  %%-difference vs 2D-solid  (all nonzero Timoshenko terms, OML reference)" % title)
    out.append("# couplings: C14=EA-GJ C15=EA-EI2 C16=EA-EI3 C23=GA2-GA3 C24=GA2-GJ C25=GA2-EI2 C34=GA3-GJ"
               " C35=GA3-EI2 C45=GJ-EI2 C56=EI2-EI3  (near-zero-denominator couplings shown but are %%-noise)")
    out.append("%-5s " % "r" + " ".join("%9s" % nm(i, j) for (i, j) in sig))
    for (r, S, RM, KL) in rows:
        M = (RM, KL)[mi]
        out.append("%-5.2f " % r + " ".join("%+9.2f" % (100 * (M[i, j] - S[i, j]) / S[i, j]) for (i, j) in sig))
    txt = "\n".join(out)
    open(os.path.join(VAL, fname), "w").write(txt + "\n")
    return txt


print("\n##### TABLE: RM %diff vs solid (all nonzero terms) #####")
print(write_pct("iea22_pcterr_RM.dat", 0, "RM (Reissner-Mindlin)"))
print("\n##### TABLE: KL %diff vs solid (all nonzero terms) #####")
print(write_pct("iea22_pcterr_KL.dat", 1, "KL (Kirchhoff)"))
print("\nwrote iea22_timo_full.dat, iea22_pcterr_RM.dat, iea22_pcterr_KL.dat -> validation/")
