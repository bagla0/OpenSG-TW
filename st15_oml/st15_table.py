"""st15 OML Timoshenko 6x6 table: for every NONZERO term, the solid (VABS) stiffness value plus the
% error of the three shell models (FEniCS-shell / JAX-Kirchhoff / JAX-RM)."""
import os
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
D = os.path.join(CC, "st_oml_compare", "data")
OUT = os.path.join(CC, "st15_oml", "st15_oml_table.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


S = Ld(os.path.join(D, "C6_st15_solid.txt"))
FK = Ld(os.path.join(D, "C6_st15_fenics_shell.txt"))
JK = Ld(os.path.join(D, "C6_st15_jax_kirch.txt"))
JR = Ld(os.path.join(D, "C6_st15_jax_rm.txt"))
thr = 1e-5 * abs(S[0, 0])                       # "nonzero" threshold (st15 symmetric zeros are exact 0)

with open(OUT, "w") as fh:
    fh.write("st15  OML Timoshenko 6x6  --  NONZERO terms only\n")
    fh.write("solid = FEniCS 2D-solid (VABS reference);  %% error = (model-solid)/solid for each shell model\n")
    fh.write("models: FEniCS-shell (FE-Kirchhoff) / JAX-Kirchhoff (C1-Hermite) / JAX-RM\n\n")
    fh.write("%-16s %15s | %11s %11s %11s\n" % ("term (C_ij)", "solid (VABS)", "FE-shell %", "JAX-Kirch %", "JAX-RM %"))
    fh.write("-" * 74 + "\n")
    for i in range(6):
        for j in range(i + 1):
            if abs(S[i, j]) < thr:
                continue
            nm = "%s (C%d%d)" % (lab[i], i + 1, i + 1) if i == j else "C%d%d %s-%s" % (j + 1, i + 1, lab[j], lab[i])
            row = "%-16s %15.4e |" % (nm, S[i, j])
            for M in (FK, JK, JR):
                e = (100 * (M[i, j] - S[i, j]) / abs(S[i, j])) if M is not None else np.nan
                row += " %10.1f%%" % e
            fh.write(row + "\n")
print("wrote", OUT)
