"""Benchmark / regression test: BAR-URC station-15 (thick-web blade station) -- RM & KL shell vs VABS.

Reads the 1-D shell SG YAML and computes the Reissner-Mindlin and Kirchhoff-Love Timoshenko 6x6,
benchmarked on the FULL 6x6 (every non-zero Cij, not just the diagonal) against the VABS .K reference.
A `max/200` cutoff drops the near-structural-zero couplings (e.g. C15, ~0.001x the largest term, whose
%-error is meaningless) and keeps the genuine terms. At this single-cell station RM and KL are close;
both track VABS to within ~18% on every kept term (the worst being the edgewise-bending block EI3 / C56,
the known thick-web shell drift).

Run:  pytest tests/test_st15_benchmark.py   |   python tests/test_st15_benchmark.py
"""
import os
import sys
import numpy as np
import pytest

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax.timo_report import sym, full_pcterr, nonzero_terms, compare_terms, print_6x6

D = os.path.join(CC, "examples", "data")
SHELL = os.path.join(D, "1d_yaml", "st15_shell.yaml")
KFILE = os.path.join(D, "benchmark", "st15_vabs.K")
TOL = 18.0          # |%err| bound on every kept non-zero term, for BOTH RM and KL
CUTOFF = 200.0      # treat |S_ij| < max/CUTOFF as a structural zero (skip)


def _parse_vabs(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if "Timoshenko Stiffness Matrix" in l)
    rows = []
    for l in lines[i + 1:]:
        q = l.split()
        if len(q) == 6:
            try: rows.append([float(x) for x in q])
            except ValueError: continue
        if len(rows) == 6: break
    return np.array(rows)


def _compute():
    from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
    from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
    S = sym(_parse_vabs(KFILE))
    RM = sym(rm_timoshenko_6x6(SHELL, 0.0, dshift=None, curved=False, shear="mitc", orient=False))
    KL = sym(gradient_junction_kirchhoff(SHELL, frac=0.0, dshift=None)[0])
    return S, RM, KL


@pytest.mark.skipif(not (os.path.exists(SHELL) and os.path.exists(KFILE)),
                    reason="station-15 data not present")
def test_st15_full6x6():
    S, RM, KL = _compute()
    eRM, eKL = full_pcterr(RM, S), full_pcterr(KL, S)
    for i, j, tag in nonzero_terms(S, CUTOFF):
        assert RM[i, i] != 0.0 or i != j
        assert abs(eRM[i, j]) < TOL, "RM %s off %.2f%% (> %.0f)" % (tag, eRM[i, j], TOL)
        assert abs(eKL[i, j]) < TOL, "KL %s off %.2f%% (> %.0f)" % (tag, eKL[i, j], TOL)
    for i in (1, 2):                                        # RM no worse than KL on the transverse shear
        assert abs(eRM[i, i]) <= abs(eKL[i, i]) + 1.0, "RM worse than KL on diagonal %d" % i


def main():
    S, RM, KL = _compute()
    print_6x6(RM, "RM (1-D shell)"); print()
    print_6x6(KL, "KL (1-D shell)")
    print("\nBAR-URC station-15 -- RM and KL vs VABS .K, every non-zero Cij term:")
    compare_terms(S, {"RM": RM, "KL": KL})


if __name__ == "__main__":
    main()
