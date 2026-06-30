"""Curved-wall multi-cell ISOTROPIC validation: 2-cell tube, JAX-KL (gradient
junction) + JAX-RM (curved, k22 from geometry) vs FEniCS-2D-solid, THIN and THICK.
Centric via dshift=t/2 (ABD shift only -- NO node shifting).  Order [EA,GA2,GA3,GJ,EI2,EI3]."""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)
from gradient_kirchhoff import gradient_junction_kirchhoff
from strip_RM import rm_timoshenko_6x6

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\multicell_tube\data"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def L(n):
    M = np.loadtxt(os.path.join(DATA, n))
    return 0.5 * (M + M.T)


def pe(m, s):
    return 100.0 * (m - s) / s


out = []
out.append("=" * 84)
out.append("2-CELL CURVED TUBE (isotropic, R=5cm, web)  --  %-error vs FEniCS-2D-solid (centric)")
out.append("JAX-KL = gradient-junction Kirchhoff ;  JAX-RM = curved RM.  No node shifting.")
out.append("=" * 84)
for tag, t, hr in (("thin", 0.004, 0.08), ("thick", 0.016, 0.32)):
    S = L("C6_solid_tube2cell_%s.txt" % tag)
    mesh = os.path.join(DATA, "tube2cell_%s.yaml" % tag)
    KF, nj, ng = gradient_junction_kirchhoff(mesh, frac=0.0, dshift=t / 2.0)
    KF = 0.5 * (np.asarray(KF) + np.asarray(KF).T)
    RM = np.asarray(rm_timoshenko_6x6(mesh, 0.0, dshift=t / 2.0, curved=True))
    RM = 0.5 * (RM + RM.T)
    out.append("\n--- %s  (h/R = %.2f, %d junctions) ---" % (tag.upper(), hr, nj))
    out.append("%-5s %14s %10s %10s" % ("term", "FE-solid", "JAX-KL %", "JAX-RM %"))
    for i in range(6):
        out.append("%-5s %14.4e %+9.1f %+9.1f" % (LBL[i], S[i, i], pe(KF[i, i], S[i, i]), pe(RM[i, i], S[i, i])))
    sk = max(abs(pe(KF[i, i], S[i, i])) for i in (1, 2))
    sr = max(abs(pe(RM[i, i], S[i, i])) for i in (1, 2))
    mk = max(abs(pe(KF[i, i], S[i, i])) for i in range(6))
    mr = max(abs(pe(RM[i, i], S[i, i])) for i in range(6))
    out.append("  max|shear(GA2,GA3)|:  KL %.1f / RM %.1f  -> %s better ;  max|all|: KL %.1f / RM %.1f"
               % (sk, sr, "RM" if sr < sk else "KL", mk, mr))

txt = "\n".join(out)
print(txt)
open(os.path.join(os.path.dirname(DATA), "tube2cell_RM_vs_KL.txt"), "w").write(txt + "\n")
