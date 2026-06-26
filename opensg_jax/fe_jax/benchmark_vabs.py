"""Benchmark: JAX 2D-solid Timoshenko 6x6 (segment.py + solid_timo.py, the Beam_solid pure-JAX pipeline)
vs a VABS `.K` reference (the Timoshenko Stiffness Matrix). FULL 6x6 comparison (every term). NO FEniCS.

  python -m opensg_jax.fe_jax.benchmark_vabs <solid_yaml> <vabs.sg.K>

%-error rule (kept identical for every Timo comparison): terms with |VABS value| < CUTOFF (1e6, denominator
noise from near-zero couplings) report 0.0% so tiny terms do not show spurious huge percentages.
"""
import sys
import os
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
else:
    from .solid_timo import compute_timo_from_yaml

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
CUTOFF = 1e6


def parse_vabs_timo(path):
    """Read the 6x6 'Timoshenko Stiffness Matrix' (order 1-ext;2,3-shear;4-twist;5,6-bend) from a VABS .K."""
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if "Timoshenko Stiffness Matrix" in l)
    rows = []
    for l in lines[i + 1:]:
        p = l.split()
        if len(p) == 6:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                continue
        if len(rows) == 6:
            break
    return np.array(rows)


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def pcterr(a, b):
    """%-error of a vs reference b; 0.0 when |b| < CUTOFF (denominator noise)."""
    return 0.0 if abs(b) < CUTOFF else 100.0 * (a - b) / b


def show(M, title):
    print("\n%s:" % title)
    for i in range(6):
        print("  " + " ".join("% .4e" % M[i, j] for j in range(6)))


def main():
    yaml_path, kpath = sys.argv[1], sys.argv[2]
    jax6 = sym(compute_timo_from_yaml(yaml_path))
    vabs6 = sym(parse_vabs_timo(kpath))
    show(jax6, "JAX (Beam_solid pipeline + 2D-YAML)")
    show(vabs6, "VABS .K reference")

    # every term of the upper triangle (diagonals + all couplings), relative to the term itself
    # (rel%) and to EA (rel_EA%, the noise-floor view -- a tiny coupling with a large rel% but a
    # negligible rel_EA% is structurally ~0, not an error).
    EA = abs(vabs6[0, 0])
    print("\n  all 6x6 terms (upper triangle: diagonals + couplings):")
    print("  %-8s %15s %15s %11s %8s %9s" % ("term", "JAX", "VABS", "abs_diff", "rel%", "rel_EA%"))
    worst_rel = worst_rel_lbl = None
    for i in range(6):
        for j in range(i, 6):
            jv, vv, ad = jax6[i, j], vabs6[i, j], abs(jax6[i, j] - vabs6[i, j])
            rel = 100.0 * ad / abs(vv) if vv != 0 else 0.0
            rel_ea = 100.0 * ad / EA
            lbl = "%s-%s" % (LBL[i], LBL[j])
            print("  %-8s %+15.6e %+15.6e %11.3e %7.4f %8.5f" % (lbl, jv, vv, ad, rel, rel_ea))
            if abs(vv) >= CUTOFF and (worst_rel is None or rel > worst_rel):
                worst_rel, worst_rel_lbl = rel, lbl

    dmax = max(100.0 * abs(jax6[i, i] - vabs6[i, i]) / abs(vabs6[i, i]) for i in range(6))
    relea_max = max(100.0 * abs(jax6[i, j] - vabs6[i, j]) / EA for i in range(6) for j in range(i, 6))
    print("\n  max |diagonal rel%%|                       = %.5f %%" % dmax)
    print("  worst coupling rel%% (|VABS term| >= 1e6)  = %.5f %% (%s)" % (worst_rel, worst_rel_lbl))
    print("  max |any term| / EA                       = %.6f %%" % relea_max)


if __name__ == "__main__":
    main()
