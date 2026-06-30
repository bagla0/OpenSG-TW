"""Section 3.1.4 tube: HIGHLY REFINED shell mesh (N=3200) with k22=0 (faceted
geometry, curvature carried by the discrete mesh -- consistent with the airfoil
cases).  Compare JAX-Kirchhoff (KL) and JAX-RM Timoshenko 6x6 to the FEniCS-2D-solid.
Centric (mid-wall) reference.  Order [EA, GA2, GA3, GJ, EI2, EI3]."""
import os
import sys
import numpy as np

TUBE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_45_45\scripts"
sys.path.insert(0, TUBE)
import tube_lib as T
from gen_meshes import gen_tube_yaml, ANI

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\data"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
N = 3200
R_REF = 0.0715
D_SHIFT = 0.008682 / 2.0
LAYUP = [(-45.0, 0.008682)]
MODE = sys.argv[1] if len(sys.argv) > 1 else "zero"     # 'zero' (k22=0) or 'exact' (k22=-1/R)
KLAB = "0" if MODE == "zero" else "-1/R"


def pe(m, s):
    return 100.0 * (m - s) / s


S = np.loadtxt(os.path.join(DATA, "C6_solid_314.txt"))
S = 0.5 * (S + S.T)

mesh = os.path.join(DATA, "shell_center_n%d.yaml" % N)
gen_tube_yaml(mesh, R_REF, layup=LAYUP, mat=ANI, n=N, ccw=True)
RM, KF = T.homog(mesh, R_REF, D_SHIFT, k22_mode=MODE)
RM = 0.5 * (RM + RM.T)
KF = 0.5 * (KF + KF.T)
np.savetxt(os.path.join(DATA, "C6_jax_kirch_center_k22%s.txt" % MODE), KF,
           header="JAX-Kirchhoff [-45] tube N=%d k22=%s centric  [EA GA2 GA3 GJ EI2 EI3]" % (N, KLAB))
np.savetxt(os.path.join(DATA, "C6_jax_rm_center_k22%s.txt" % MODE), RM,
           header="JAX-RM [-45] tube N=%d k22=%s centric  [EA GA2 GA3 GJ EI2 EI3]" % (N, KLAB))

out = []
out.append("=" * 92)
out.append("Section 3.1.4 tube  R=7.15cm t=8.682mm [-45]  |  refined shell N=%d, k22=%s, centric ref" % (N, KLAB))
out.append("TIMOSHENKO 6x6  [EA, GA2, GA3, GJ, EI2, EI3]   EA,GA[N]  GJ,EI[N.m^2]")
out.append("=" * 92)
out.append("%-6s %14s %14s %8s %14s %8s %9s" % ("term", "FE-solid", "JAX-KL", "KL %", "JAX-RM", "RM %", "KL-RM %"))
for i in range(6):
    klrm = 100.0 * (KF[i, i] - RM[i, i]) / RM[i, i]
    out.append("%-6s %14.4e %14.4e %+7.2f %14.4e %+7.2f %+8.2f"
               % (LBL[i], S[i, i], KF[i, i], pe(KF[i, i], S[i, i]),
                  RM[i, i], pe(RM[i, i], S[i, i]), klrm))
mk = max(abs(pe(KF[i, i], S[i, i])) for i in range(6))
mr = max(abs(pe(RM[i, i], S[i, i])) for i in range(6))
out.append("max|diag err vs solid|:   KL %.2f%%    RM %.2f%%" % (mk, mr))
out.append("GA2/GA3 split:   KL %.3f%%    RM %.3f%%   (axisymmetry check; 0 = exact)"
           % (100 * (KF[1, 1] - KF[2, 2]) / KF[1, 1], 100 * (RM[1, 1] - RM[2, 2]) / RM[1, 1]))

# dominant couplings
dmax = np.max(np.abs(np.diag(S)))
coup = sorted([(i, j) for i in range(6) for j in range(i) if abs(S[i, j]) > 1e-3 * dmax],
              key=lambda ij: -abs(S[ij[0], ij[1]]))
out.append("\nDominant couplings (C_{j+1,i+1}):")
out.append("%-14s %14s %14s %8s %14s %8s" % ("coupling", "FE-solid", "JAX-KL", "KL %", "JAX-RM", "RM %"))
for i, j in coup:
    out.append("%-14s %14.4e %14.4e %+7.2f %14.4e %+7.2f"
               % ("C%d%d %s-%s" % (j + 1, i + 1, LBL[j], LBL[i]),
                  S[i, j], KF[i, j], pe(KF[i, j], S[i, j]), RM[i, j], pe(RM[i, j], S[i, j])))

txt = "\n".join(out)
print(txt)
open(os.path.join(DATA, "timo_k22%s_N%d.txt" % (MODE, N)), "w").write(txt + "\n")
print("\nwrote timo_k22%s_N%d.txt" % (MODE, N))
