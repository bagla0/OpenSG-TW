"""Example 1 - RM (Reissner-Mindlin) Timoshenko 6x6 from a 1-D shell YAML.

Reproduces the RM tutorial from the command line: loads the [-45] tube shell SG YAML, emits the
e1/e2/e3 orientation PNG, computes the RM 6x6 (MITC transverse shear), and prints the per-term
%-error against the 2-D solid reference (and against KL, to show the shear recovery).

Run (Windows, env on PATH):
    python examples/1_get_beam_props_rm_shell.py
"""
import os
import sys
import numpy as np

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "rm", "opensg_jax", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4, linewidth=150, suppress=True)

from fe_jax.orient_plot import plot_orient
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
SHELL = os.path.join(CC, "tube_thesis_314", "data", "shell_center.yaml")
SOLIDY = os.path.join(CC, "tube_thesis_314", "data", "solid_m45.yaml")
BENCH = os.path.join(CC, "tube_thesis_314", "data", "C6_solid_314.txt")


def sym(M):
    M = np.asarray(M, dtype=float); return 0.5 * (M + M.T)


def main():
    print("orientation ->", plot_orient(SHELL, SOLIDY))
    C_rm = sym(rm_timoshenko_6x6(SHELL, 0.0, dshift=None, curved=True, shear="mitc"))
    C_kl = sym(gradient_junction_kirchhoff(SHELL, frac=0.0, dshift=None)[0])
    S = sym(np.loadtxt(BENCH))

    print("\nRM Timoshenko 6x6 [EA, GA2, GA3, GJ, EI2, EI3]:")
    print(C_rm)
    print("\n  %-5s %14s %14s %14s %9s %9s" % ("term", "RM", "KL", "solid", "RM%err", "KL%err"))
    for i in range(6):
        rm = 100 * (C_rm[i, i] - S[i, i]) / S[i, i]
        kl = 100 * (C_kl[i, i] - S[i, i]) / S[i, i]
        print("  %-5s %14.5e %14.5e %14.5e %+8.2f %+8.2f"
              % (LBL[i], C_rm[i, i], C_kl[i, i], S[i, i], rm, kl))


if __name__ == "__main__":
    main()
