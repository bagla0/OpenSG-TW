"""Example 3 - JAX 2-D solid Timoshenko 6x6 from a 2-D solid YAML.

Loads the MH-104 airfoil 2-D solid SG YAML, emits the e1/e2/e3 orientation PNG (solid + shell), computes
the full Timoshenko 6x6 with the pure-JAX MSG solver, and prints the per-term %-error -- diagonals AND
every coupling -- against the VABS .K reference.

Run:
    python examples/3_get_beam_props_jax_solid.py
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
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
SOLIDY = os.path.join(CC, "examples", "data", "2d_yaml", "mh104_solid.yaml")
SHELL = os.path.join(CC, "examples", "data", "1d_yaml", "mh104_shell.yaml")
KFILE = os.path.join(CC, "examples", "data", "benchmark", "mh104.sg.K")


def sym(M):
    M = np.asarray(M, dtype=float); return 0.5 * (M + M.T)


def parse_vabs(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if "Timoshenko Stiffness Matrix" in l)
    rows = []
    for l in lines[i + 1:]:
        q = l.split()
        if len(q) == 6:
            try:
                rows.append([float(x) for x in q])
            except ValueError:
                continue
        if len(rows) == 6:
            break
    return np.array(rows)


def main():
    try:
        print("orientation ->", plot_orient(SHELL, SOLIDY))
    except Exception as e:
        print("orientation skipped:", e)
    C6 = sym(compute_timo_from_yaml(SOLIDY, verbose=False))
    V = sym(parse_vabs(KFILE))

    print("\nJAX-solid Timoshenko 6x6 [EA, GA2, GA3, GJ, EI2, EI3]:")
    print(C6)
    print("\n  %-5s %15s %15s %12s" % ("term", "JAX-solid", "VABS", "%err"))
    for i in range(6):
        e = 100 * (C6[i, i] - V[i, i]) / V[i, i]
        print("  %-5s %15.5e %15.5e %+11.4f" % (LBL[i], C6[i, i], V[i, i], e))
    worst = max(abs(100 * (C6[i, j] - V[i, j]) / V[i, j])
                for i in range(6) for j in range(6) if abs(V[i, j]) > 1e6)
    print("\nworst term with |VABS|>=1e6 : %.4f %%" % worst)


if __name__ == "__main__":
    main()
