"""Regenerate oml_mh104/data/oml_timo_table.txt with the OML diagonal for all three shell models
(JAX-Kirchhoff, JAX-RM, FEniCS-shell) vs FEniCS solid + the % errors, for f=0.1..0.6."""
import os
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
RES = os.path.join(CC, "mh104_thickness_study", "results")
OUT = os.path.join(CC, "oml_mh104", "data", "oml_timo_table.txt")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
FS = [10, 20, 40, 60]


def vals(tag, C):
    return "  %-12s " % tag + "  ".join("%s=%.4e" % (lab[i], C[i, i]) for i in range(6)) + "\n"


def errs(tag, C, S):
    return "  %-12s " % tag + "  ".join("%s=%+7.1f%%" % (lab[i], 100 * (C[i, i] - S[i, i]) / abs(S[i, i])) for i in range(6)) + "\n"


with open(OUT, "w") as fh:
    fh.write("mh104 OML Timoshenko DIAGONAL: JAX-Kirchhoff / JAX-RM / FEniCS-shell vs FEniCS solid\n")
    fh.write("order [EA,GA2,GA3,GJ,EI2,EI3]   (f=0.1..0.6)\n\n")
    for fi in FS:
        K = np.loadtxt(os.path.join(RES, "C6_shell_jax_OML_f%03d.txt" % fi))
        R = np.loadtxt(os.path.join(RES, "C6_shell_rm_OML_f%03d.txt" % fi))
        F = np.loadtxt(os.path.join(RES, "C6_fenics_shell_OML_f%03d.txt" % fi))
        S = np.loadtxt(os.path.join(RES, "C6_solid_f%03d.txt" % fi))
        fh.write("== f=%.2f ==\n" % (fi / 100))
        fh.write(vals("JAX-Kirchhoff", K)); fh.write(vals("JAX-RM", R))
        fh.write(vals("FEniCS-shell", F)); fh.write(vals("FEniCS-solid", S))
        fh.write(errs("Kirch %err", K, S)); fh.write(errs("RM %err", R, S))
        fh.write(errs("FE-shell %err", F, S)); fh.write("\n")
print("wrote", OUT)
