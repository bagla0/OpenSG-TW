"""Benchmark: JAX 2D-solid Timoshenko 6x6 (segment.py + solid_timo.py, the Beam_solid pure-JAX pipeline)
vs a VABS `.K` reference (the Timoshenko Stiffness Matrix). FULL 6x6 comparison (every term). NO FEniCS.

  python -m opensg_jax.fe_jax.benchmark_vabs <solid_yaml> <vabs.sg.K>

%-error rule (kept identical for every Timo comparison): use ALL nonzero terms, but NEGLECT any term
whose |value| is >= 1000x below the largest |term| in the reference matrix (i.e. |term| < max|term|/1000) --
those are denominator noise. The neglect threshold scales with the matrix, no fixed cutoff.
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
NEGLECT_RATIO = 1000.0   # neglect terms >= this many times below the max |term|


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

    thresh = float(np.max(np.abs(vabs6))) / NEGLECT_RATIO
    print("\n  neglect threshold = max|VABS| / %g = %.4e   (terms below -> neglected)" % (NEGLECT_RATIO, thresh))
    print("\n  all nonzero 6x6 terms (upper triangle: diagonals + couplings):")
    print("  %-8s %16s %16s %11s %9s  %s" % ("term", "JAX", "VABS", "abs_diff", "rel%", "kept?"))
    worst = worst_lbl = None
    for i in range(6):
        for j in range(i, 6):
            jv, vv = jax6[i, j], vabs6[i, j]; ad = abs(jv - vv)
            keep = abs(vv) >= thresh
            rel = 100.0 * ad / abs(vv) if vv != 0 else 0.0
            lbl = "%s-%s" % (LBL[i], LBL[j])
            print("  %-8s %+16.6e %+16.6e %11.3e %9s  %s"
                  % (lbl, jv, vv, ad, ("%.4f" % rel) if keep else "   -   ", "yes" if keep else "neglect"))
            if keep and (worst is None or rel > worst):
                worst, worst_lbl = rel, lbl

    dmax = max(100.0 * abs(jax6[i, i] - vabs6[i, i]) / abs(vabs6[i, i]) for i in range(6))
    print("\n  max |diagonal rel%%|                            = %.5f %%" % dmax)
    print("  worst rel%% over kept terms (|term|>=max/%g)  = %.5f %% (%s)" % (NEGLECT_RATIO, worst, worst_lbl))


if __name__ == "__main__":
    main()
