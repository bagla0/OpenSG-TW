"""Thesis Section 3.1.4 circular-tube benchmark.

Single-ply [-45 deg] orthotropic tube:
    mean radius  R = 7.15 cm  = 0.0715 m
    wall thick   t = 8.682 mm = 0.008682 m   (single ply)
    material ud_frp  E1=37 E2=E3=9 GPa  G12=G13=G23=4 GPa  nu=0.28

Thesis Table 3.4 reference (MSG-solid = 3D-solid FE, reference-invariant):
    C1b1 (EA)          = 4.7738e7  N
    C1b2 (ext-twist)   = -9.3906e5 N.m
    C2b2 (GJ)          = 1.4932e5  N.m^2
    C3b3 = C4b4 (EI)   = 1.0741e5  N.m^2

Computes JAX-Kirchhoff (KL) and JAX-RM at OML/center/IML, reduces the 6x6
Timoshenko to the 4x4 Euler-Bernoulli [EA, GJ, EI2, EI3], and reports the
%-error of each shell model vs the thesis solid.  Centric (center) is the
thesis MSG-TW reference and the headline comparison.
"""
import os
import sys
import numpy as np

TUBE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_45_45\scripts"
sys.path.insert(0, TUBE)
import tube_lib as T
from gen_meshes import gen_tube_yaml, ANI

HERE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314"
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

R_MEAN = 0.0715          # 7.15 cm mean radius
T_WALL = 0.008682        # 8.682 mm single ply
R_OUT = R_MEAN + T_WALL / 2.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3200   # highly refined shell mesh
LAYUP = [(-45.0, T_WALL)]  # single -45 ply

# ref -> (R_ref, d_shift)   d_shift = JAX shift_abd_reference distance from OML
REFS = {"OML":    (R_OUT,                 0.0),
        "center": (R_OUT - T_WALL / 2.0,  T_WALL / 2.0),
        "IML":    (R_OUT - T_WALL,        T_WALL)}

# thesis Table 3.4 (MSG-solid), EB 4x4 order [EA, ext-twist(EA-GJ), GJ, EI]
SOLID = {"EA": 4.7738e7, "ext": -9.3906e5, "GJ": 1.4932e5, "EI": 1.0741e5}


def eb_from_timo(C6):
    """Reduce the 6x6 Timoshenko [EA,GA2,GA3,GJ,EI2,EI3] to the 4x4 EB
    [EA,GJ,EI2,EI3] by static condensation of the two shear DOFs."""
    S = np.linalg.inv(C6)
    keep = [0, 3, 4, 5]
    return np.linalg.inv(S[np.ix_(keep, keep)])  # [EA, GJ, EI2, EI3]


def terms(eb):
    # EA, ext-twist coupling, GJ, EI2, EI3
    return eb[0, 0], eb[0, 1], eb[1, 1], eb[2, 2], eb[3, 3]


def pe(v, ref):
    return 100.0 * (v - ref) / ref


results = {}
for ref, (R_ref, d_shift) in REFS.items():
    mesh = os.path.join(DATA, "shell_%s.yaml" % ref)
    gen_tube_yaml(mesh, R_ref, layup=LAYUP, mat=ANI, n=N, ccw=True)
    RM, KF = T.homog(mesh, R_ref, d_shift, k22_mode="exact")
    RM = 0.5 * (RM + RM.T)
    KF = 0.5 * (KF + KF.T)
    np.savetxt(os.path.join(DATA, "C6_jax_kirch_%s.txt" % ref), KF,
               header="JAX-Kirchhoff [-45] tube R=%.5f ref=%s order [EA GA2 GA3 GJ EI2 EI3]" % (R_ref, ref))
    np.savetxt(os.path.join(DATA, "C6_jax_rm_%s.txt" % ref), RM,
               header="JAX-RM [-45] tube R=%.5f ref=%s order [EA GA2 GA3 GJ EI2 EI3]" % (R_ref, ref))
    results[ref] = (KF, RM, eb_from_timo(KF), eb_from_timo(RM))

lines = []
lines.append("Section 3.1.4 circular tube : R=7.15cm  t=8.682mm  single ply [-45]  h/R=%.4f" % (T_WALL / R_MEAN))
lines.append("Euler-Bernoulli 4x4 reduced from the JAX Timoshenko 6x6.  Units: EA[N] ext[N.m] GJ,EI[N.m^2]")
lines.append("")
lines.append("THESIS MSG-solid (Table 3.4, reference):")
lines.append("   EA = %12.4e   ext-twist = %12.4e   GJ = %12.4e   EI = %12.4e"
             % (SOLID["EA"], SOLID["ext"], SOLID["GJ"], SOLID["EI"]))
lines.append("")
for ref in ("OML", "center", "IML"):
    KF, RM, ebK, ebR = results[ref]
    tag = "  <-- thesis MSG-TW Centric" if ref == "center" else ""
    lines.append("=== reference surface = %s ===%s" % (ref, tag))
    lines.append("%-7s %12s %12s %12s %12s | %8s %8s %8s %8s"
                 % ("model", "EA", "ext-twist", "GJ", "EI", "EA%", "ext%", "GJ%", "EI%"))
    for nm, eb in (("JAX-KL", ebK), ("JAX-RM", ebR)):
        EA, ext, GJ, EI2, EI3 = terms(eb)
        EI = 0.5 * (EI2 + EI3)
        lines.append("%-7s %12.4e %12.4e %12.4e %12.4e | %+7.2f %+7.2f %+7.2f %+7.2f"
                     % (nm, EA, ext, GJ, EI,
                        pe(EA, SOLID["EA"]), pe(ext, SOLID["ext"]),
                        pe(GJ, SOLID["GJ"]), pe(EI, SOLID["EI"])))
    lines.append("")

txt = "\n".join(lines)
print(txt)
open(os.path.join(DATA, "comparison_314.txt"), "w").write(txt + "\n")
print("wrote", os.path.join(DATA, "comparison_314.txt"))
