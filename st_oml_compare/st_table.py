"""Table (.txt): st15 & st12 OML Timoshenko 6x6, % error of the three shell models
(FEniCS-shell = FEniCS-Kirchhoff, JAX-Kirchhoff, JAX-RM) vs the FEniCS 2D-solid (VABS reference),
for EVERY nonzero term."""
import os
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
D = os.path.join(CC, "st_oml_compare", "data")
OUT = os.path.join(CC, "st_oml_compare", "st_comparison_table.txt")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


with open(OUT, "w") as fh:
    fh.write("st15 / st12  OML Timoshenko 6x6:  % ERROR vs FEniCS 2D-solid (VABS reference)\n")
    fh.write("models: FEniCS-shell (FE-Kirchhoff) / JAX-Kirchhoff / JAX-RM.   order [EA,GA2,GA3,GJ,EI2,EI3]\n")
    for st in ("15", "12"):
        S = Ld(os.path.join(D, "C6_st%s_solid.txt" % st))
        FK = Ld(os.path.join(D, "C6_st%s_fenics_shell.txt" % st))
        JK = Ld(os.path.join(D, "C6_st%s_jax_kirch.txt" % st))
        JR = Ld(os.path.join(D, "C6_st%s_jax_rm.txt" % st))
        fh.write("\n================  st%s  ================\n" % st)
        if S is None:
            fh.write("  (solid reference missing)\n"); continue
        fh.write("%-15s %13s | %10s %10s %10s\n" % ("term", "solid(VABS)", "FE-shell%", "JAX-Kir%", "JAX-RM%"))
        for i in range(6):
            for j in range(i + 1):
                nm = lab[i] if i == j else "C%d%d %s-%s" % (j + 1, i + 1, lab[j], lab[i])
                row = "%-15s %13.4e |" % (nm, S[i, j])
                for M in (FK, JK, JR):
                    e = (100 * (M[i, j] - S[i, j]) / abs(S[i, j])) if (M is not None and abs(S[i, j]) > 0) else np.nan
                    row += " %9.1f%%" % e
                fh.write(row + "\n")
print("wrote", OUT)
