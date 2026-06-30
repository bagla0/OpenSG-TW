"""Timoshenko 6x6 comparison for the Section 3.1.4 single-ply [-45] tube.
FEniCS-2D-solid (reference) vs JAX-Kirchhoff (KL) and JAX-RM, at OML/center/IML.
Order [EA, GA2, GA3, GJ, EI2, EI3]."""
import os
import numpy as np

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\data"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
THESIS_EB = {"EA": 4.7738e7, "ext": -9.3906e5, "GJ": 1.4932e5, "EI": 1.0741e5}


def L(name):
    M = np.loadtxt(os.path.join(DATA, name))
    return 0.5 * (M + M.T)


def pe(m, s):
    return 100.0 * (m - s) / s


def eb(C6):
    S = np.linalg.inv(C6)
    return np.linalg.inv(S[np.ix_([0, 3, 4, 5], [0, 3, 4, 5])])  # [EA,GJ,EI2,EI3]


S = L("C6_solid_314.txt")
REF = {r: (L("C6_jax_kirch_%s.txt" % r), L("C6_jax_rm_%s.txt" % r))
       for r in ("OML", "center", "IML")}

out = []
out.append("=" * 96)
out.append("Section 3.1.4 tube  R=7.15cm  t=8.682mm  single ply [-45]  h/R=0.121")
out.append("TIMOSHENKO 6x6  order [EA, GA2, GA3, GJ, EI2, EI3]   units: EA,GA[N]  GJ,EI[N.m^2]")
out.append("=" * 96)

out.append("\nFEniCS-2D-solid Timoshenko 6x6 (reference, 8800 tris):")
for i in range(6):
    out.append("  " + "  ".join("%11.4e" % S[i, j] for j in range(6)))

# sanity: reduce solid 6x6 -> EB and check vs thesis Table 3.4
ebS = eb(S)
out.append("\nSolid -> Euler-Bernoulli 4x4 reduction vs thesis Table 3.4 (MSG-solid):")
out.append("  EA   solid=%.4e  thesis=%.4e  (%+.2f%%)" % (ebS[0, 0], THESIS_EB["EA"], pe(ebS[0, 0], THESIS_EB["EA"])))
out.append("  ext  solid=%.4e  thesis=%.4e  (%+.2f%%)" % (ebS[0, 1], THESIS_EB["ext"], pe(ebS[0, 1], THESIS_EB["ext"])))
out.append("  GJ   solid=%.4e  thesis=%.4e  (%+.2f%%)" % (ebS[1, 1], THESIS_EB["GJ"], pe(ebS[1, 1], THESIS_EB["GJ"])))
out.append("  EI   solid=%.4e  thesis=%.4e  (%+.2f%%)" % (ebS[2, 2], THESIS_EB["EI"], pe(ebS[2, 2], THESIS_EB["EI"])))

# headline: diagonal %err at centric reference
K, R = REF["center"]
out.append("\n" + "-" * 96)
out.append("TIMOSHENKO DIAGONAL %err vs solid  @ centric (mid-wall) reference  [thesis MSG-TW surface]")
out.append("-" * 96)
out.append("%-5s %14s %14s %9s %14s %9s" % ("term", "solid", "JAX-KL", "KL %", "JAX-RM", "RM %"))
for i in range(6):
    out.append("%-5s %14.4e %14.4e %+8.2f %14.4e %+8.2f"
               % (LBL[i], S[i, i], K[i, i], pe(K[i, i], S[i, i]), R[i, i], pe(R[i, i], S[i, i])))
mk = max(abs(pe(K[i, i], S[i, i])) for i in range(6))
mr = max(abs(pe(R[i, i], S[i, i])) for i in range(6))
out.append("max|diag err|:  KL %.2f%%   RM %.2f%%   -> %s better overall" % (mk, mr, "RM" if mr < mk else "KL"))

# dominant couplings
dmax = np.max(np.abs(np.diag(S)))
coup = sorted([(i, j) for i in range(6) for j in range(i) if abs(S[i, j]) > 1e-3 * dmax],
              key=lambda ij: -abs(S[ij[0], ij[1]]))
out.append("\nDominant couplings  @ centric reference  (C_{j+1,i+1}):")
out.append("%-14s %14s %14s %9s %14s %9s" % ("coupling", "solid", "JAX-KL", "KL %", "JAX-RM", "RM %"))
for i, j in coup:
    nm = "C%d%d %s-%s" % (j + 1, i + 1, LBL[j], LBL[i])
    out.append("%-14s %14.4e %14.4e %+8.2f %14.4e %+8.2f"
               % (nm, S[i, j], K[i, j], pe(K[i, j], S[i, j]), R[i, j], pe(R[i, j], S[i, j])))

# transverse-shear spotlight (where RM should win)
out.append("\n" + "-" * 96)
out.append("TRANSVERSE-SHEAR spotlight (GA2, GA3) -- the Timoshenko terms absent from the EB 4x4")
out.append("-" * 96)
for r in ("OML", "center", "IML"):
    Kr, Rr = REF[r]
    out.append("%-7s  GA2: solid=%.4e  KL=%+6.2f%%  RM=%+6.2f%%   |  GA3: KL=%+6.2f%%  RM=%+6.2f%%"
               % (r, S[1, 1], pe(Kr[1, 1], S[1, 1]), pe(Rr[1, 1], S[1, 1]),
                  pe(Kr[2, 2], S[2, 2]), pe(Rr[2, 2], S[2, 2])))

# reference-effect (diagonal max err per reference)
out.append("\nReference effect (diagonal max|err|, solid is reference-invariant):")
for r in ("OML", "center", "IML"):
    Kr, Rr = REF[r]
    mkr = max(abs(pe(Kr[i, i], S[i, i])) for i in range(6))
    mrr = max(abs(pe(Rr[i, i], S[i, i])) for i in range(6))
    out.append("  %-7s  KL %5.2f%%   RM %5.2f%%" % (r, mkr, mrr))

txt = "\n".join(out)
print(txt)
open(os.path.join(DATA, "timo_comparison_314.txt"), "w").write(txt + "\n")
print("\nwrote", os.path.join(DATA, "timo_comparison_314.txt"))
