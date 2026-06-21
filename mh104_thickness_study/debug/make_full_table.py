"""Full mh104 OML Timoshenko comparison table (.txt): every nonzero 6x6 term, all four models
(JAX-Kirchhoff, JAX-RM, FEniCS-shell, FEniCS-solid) + the % error of each shell model vs solid."""
import os
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
RES = os.path.join(CC, "mh104_thickness_study", "results")
OUT = os.path.join(RES, "timo_full_comparison.txt")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
ALLF = [10, 20, 30, 40, 60, 75, 80, 100]
PRE = [("Kirchhoff", "C6_shell_jax_OML_"), ("RM", "C6_shell_rm_OML_"),
       ("FEshell", "C6_fenics_shell_OML_"), ("solid", "C6_solid_")]


def Ld(p):
    return np.loadtxt(p) if os.path.exists(p) else None


with open(OUT, "w") as fh:
    fh.write("mh104 OML Timoshenko 6x6 comparison -- JAX-Kirchhoff / JAX-RM / FEniCS-shell / FEniCS-solid\n")
    fh.write("order [EA,GA2,GA3,GJ,EI2,EI3];  C_ij = bracket index.  (blank = model not available at that f)\n")
    for fi in ALLF:
        M = {nm: Ld(os.path.join(RES, "%sf%03d.txt" % (pre, fi))) for nm, pre in PRE}
        if M["solid"] is None:
            continue
        S = M["solid"]
        fh.write("\n================  f = %.2f  ================\n" % (fi / 100))
        fh.write("%-14s %12s %12s %12s %12s | %8s %8s %8s\n" %
                 ("term", "Kirchhoff", "RM", "FEshell", "solid", "Kir%", "RM%", "FE%"))
        for i in range(6):
            for j in range(i + 1):
                nm = lab[i] if i == j else "C%d%d %s-%s" % (j + 1, i + 1, lab[j], lab[i])
                row = "%-14s" % nm
                for k in ("Kirchhoff", "RM", "FEshell", "solid"):
                    row += " %12.4e" % (M[k][i, j] if M[k] is not None else np.nan)
                row += " |"
                for k in ("Kirchhoff", "RM", "FEshell"):
                    e = (100 * (M[k][i, j] - S[i, j]) / abs(S[i, j])) if (M[k] is not None and abs(S[i, j]) > 0) else np.nan
                    row += " %7.1f%%" % e
                fh.write(row + "\n")
print("wrote", OUT)
