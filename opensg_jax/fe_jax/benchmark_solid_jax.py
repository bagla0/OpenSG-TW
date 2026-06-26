"""Benchmark: JAX 2D-solid Timoshenko 6x6 (segment.py + solid_timo.py) vs the FEniCS solid reference.

  python -m opensg_jax.fe_jax.benchmark_solid_jax <solid_yaml> <fenics_C6.txt>

Prints both 6x6 matrices and the per-term %-error (terms with |value| < 1e6 -> 0, denominator noise).
"""
import sys
import numpy as np

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
else:
    from .solid_timo import compute_timo_from_yaml

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def show(M, title):
    print("\n%s:" % title)
    for i in range(6):
        print("  " + " ".join("% .4e" % M[i, j] for j in range(6)))


def main():
    yaml_path, ref_path = sys.argv[1], sys.argv[2]
    jax6 = sym(compute_timo_from_yaml(yaml_path))
    fe6 = sym(np.loadtxt(ref_path))

    show(jax6, "JAX 2D-solid (segment + solid_timo)")
    show(fe6, "FEniCS 2D-solid reference")

    print("\n  term   JAX            FEniCS         %err")
    for i in range(6):
        e = 100 * (jax6[i, i] - fe6[i, i]) / fe6[i, i]
        print("  %-4s  %.5e  %.5e  %+8.3f" % (LBL[i], jax6[i, i], fe6[i, i], e))
    dmax = max(abs(100 * (jax6[i, i] - fe6[i, i]) / fe6[i, i]) for i in range(6))
    print("\n  max |diagonal %% error| = %.3f %%" % dmax)

    # full-matrix relative error on non-negligible terms
    bad = 0.0
    for i in range(6):
        for j in range(6):
            if abs(fe6[i, j]) >= 1e6:
                bad = max(bad, abs(100 * (jax6[i, j] - fe6[i, j]) / fe6[i, j]))
    print("  max |6x6 %% error| (|term|>=1e6) = %.3f %%" % bad)


if __name__ == "__main__":
    main()
