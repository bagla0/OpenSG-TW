"""Example 2 - KL (Kirchhoff-Love) Timoshenko 6x6 from a 1-D shell YAML.

Loads the [-45] tube shell SG YAML, emits the e1/e2/e3 orientation PNG, computes the KL 6x6
(Hermite-C1 arc elements), and prints the per-term %-error against the 2-D solid reference. KL is
exact in the classical limit but under-predicts the transverse-shear terms GA2,GA3 (use example 1, RM,
for those).

Run:
    python examples/2_get_beam_props_kl_shell.py
"""
import os
import sys
import numpy as np

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4, linewidth=150, suppress=True)

from fe_jax.orient_plot import plot_orient
from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
SHELL = os.path.join(CC, "tests", "research", "tube_thesis_314", "data", "shell_center.yaml")
SOLIDY = os.path.join(CC, "tests", "research", "tube_thesis_314", "data", "solid_m45.yaml")
BENCH = os.path.join(CC, "tests", "research", "tube_thesis_314", "data", "C6_solid_314.txt")


def sym(M):
    M = np.asarray(M, dtype=float); return 0.5 * (M + M.T)


def main():
    print("orientation ->", plot_orient(SHELL, SOLIDY))
    C_kl = sym(gradient_junction_kirchhoff(SHELL, frac=0.0, dshift=None)[0])
    S = sym(np.loadtxt(BENCH))

    print("\nKL Timoshenko 6x6 [EA, GA2, GA3, GJ, EI2, EI3]:")
    print(C_kl)
    print("\n  %-5s %15s %15s %12s" % ("term", "KL", "solid", "%err"))
    for i in range(6):
        e = 100 * (C_kl[i, i] - S[i, i]) / S[i, i]
        print("  %-5s %15.5e %15.5e %+11.3f" % (LBL[i], C_kl[i, i], S[i, i], e))


if __name__ == "__main__":
    main()
