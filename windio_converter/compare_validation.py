"""IEA-22 windIO validation: RM & KL (1D-shell) vs FEniCS-2D-solid, ALL nonzero Timoshenko terms.
Stores C6_RM_*.txt and C6_KL_*.txt (and the solid is C6_solid_*.txt), emits a per-station solid+shell
orientation PNG, prints the analysis, and writes validation_summary.txt."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff
from fe_jax.orient_plot import plot_orient

VAL = os.path.join(CC, "windio_converter", "validation")
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
TAGS = sys.argv[1:] or ["r030", "r050", "r070"]
# windIO defines the OML (outer mold line) contour ONLY -> reference the shell ABD at the OML face
# (plies ordered outer->inner, frac=0 = first-ply/OML face). NO mid-surface offset (that was mh104-specific).
# The PreVABS solid is likewise OML-native (baseline = OML airfoil, layups build inward).
FRAC_OML = 0.0


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def pe(m, s):
    return 100.0 * (m - s) / s if abs(s) > 0 else float("nan")


def nm(i, j):
    return LBL[i] if i == j else "C%d%d %s-%s" % (i + 1, j + 1, LBL[i], LBL[j])


out = []
for tag in TAGS:
    shell = os.path.join(VAL, "shell_iea22_%s.yaml" % tag)
    solidy = os.path.join(VAL, "solid_iea22_%s.yaml" % tag)
    try:
        plot_orient(shell, solidy if os.path.exists(solidy) else None, os.path.join(VAL, "orient_iea22_%s.png" % tag))
    except Exception as e:
        print("[orient skipped %s] %s" % (tag, e))
    RM = sym(rm_timoshenko_6x6(shell, FRAC_OML, orient=False))
    KL = sym(gradient_junction_kirchhoff(shell, frac=FRAC_OML, orient=False)[0])
    np.savetxt(os.path.join(VAL, "C6_RM_iea22_%s.txt" % tag), RM,
               header="JAX MSG-RM (Reissner-Mindlin) IEA-22 %s [EA GA2 GA3 GJ EI2 EI3], frac=0 (OML)" % tag)
    np.savetxt(os.path.join(VAL, "C6_KL_iea22_%s.txt" % tag), KL,
               header="JAX MSG-KL (Kirchhoff) IEA-22 %s [EA GA2 GA3 GJ EI2 EI3], frac=0 (OML)" % tag)
    sp = os.path.join(VAL, "C6_solid_iea22_%s.txt" % tag)
    if not os.path.exists(sp):
        out.append("\n=== %s : RM/KL stored; SOLID not ready yet ===" % tag); continue
    S = sym(np.loadtxt(sp))
    out.append("\n=== IEA-22 %s : RM & KL vs 2D-solid (all nonzero terms) ===" % tag)
    out.append("  term            solid          norm     RM %err    KL %err")
    for i in range(6):
        for j in range(i, 6):
            s = S[i, j]; norm = s / np.sqrt(abs(S[i, i] * S[j, j]))
            if i == j or abs(norm) >= 0.03:
                out.append("  %-14s %+.4e  %+.3f  %+8.2f  %+8.2f" % (nm(i, j), s, norm, pe(RM[i, j], s), pe(KL[i, j], s)))
    dRM = max(abs(pe(RM[i, i], S[i, i])) for i in range(6))
    dKL = max(abs(pe(KL[i, i], S[i, i])) for i in range(6))
    out.append("  max|diagonal %%err|:  RM %.2f%%   KL %.2f%%   -> %s" % (dRM, dKL, "RM" if dRM < dKL else "KL"))

txt = "\n".join(out)
print(txt)
open(os.path.join(VAL, "validation_summary.txt"), "w").write(txt + "\n")
print("\nstored: C6_RM_iea22_*.txt, C6_KL_iea22_*.txt, validation_summary.txt, orient_iea22_*.png -> validation/")
