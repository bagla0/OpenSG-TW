"""Regenerate oml_mh104/data/oml_timo_table.txt with the 3-way OML diagonal: JAX-Kirchhoff, JAX-RM,
FEniCS solid, and the % errors of each model."""
import os
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
RES = os.path.join(CC, "mh104_thickness_study", "results")
OUT = os.path.join(CC, "oml_mh104", "data", "oml_timo_table.txt")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 40, 60, 80, 100]


def row(tag, C):
    return "  %-10s " % tag + "  ".join("%s=%.4e" % (lab[i], C[i, i]) for i in range(6)) + "\n"


def erow(tag, C, S):
    return "  %-10s " % tag + "  ".join("%s=%+6.1f%%" % (lab[i], 100 * (C[i, i] - S[i, i]) / abs(S[i, i])) for i in range(6)) + "\n"


with open(OUT, "w") as fh:
    fh.write("mh104 OML-reference Timoshenko DIAGONAL: JAX-Kirchhoff vs JAX-Reissner-Mindlin vs FEniCS solid\n")
    fh.write("order [EA,GA2,GA3,GJ,EI2,EI3]\n\n")
    for fi in FS:
        K = np.loadtxt(os.path.join(RES, "C6_shell_jax_OML_f%03d.txt" % fi))
        R = np.loadtxt(os.path.join(RES, "C6_shell_rm_OML_f%03d.txt" % fi))
        S = np.loadtxt(os.path.join(RES, "C6_solid_f%03d.txt" % fi))
        fh.write("== f=%.2f ==\n" % (fi / 100))
        fh.write(row("Kirchhoff", K)); fh.write(row("RM", R)); fh.write(row("solid", S))
        fh.write(erow("Kirch %err", K, S)); fh.write(erow("RM %err", R, S)); fh.write("\n")
print("wrote", OUT)
