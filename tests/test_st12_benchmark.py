"""Benchmark / regression test: station-12 blade cross-section -- RM & KL shell vs 2-D solid.

A thick, web-stiffened airfoil station whose bundled 2-D-solid reference is a COARSE mesh, so the
shell-vs-solid couplings are large and the solid reference is itself only indicative. The test therefore
benchmarks on the full 6x6 in two complementary ways:
  - each DIAGONAL term (EA, GA2, GA3, GJ, EI2, EI3) within a documented per-term bound of the 2-D solid
    (the thick-section / coarse-mesh gap is real and pinned, not hidden);
  - RM and KL agree on EVERY non-zero Cij term (shell self-consistency) -- the robust full-matrix check
    that does not over-trust the coarse solid couplings;
  - RM is at least as good as KL on the transverse-shear terms.

Run:  pytest tests/test_st12_benchmark.py   |   python tests/test_st12_benchmark.py
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
from opensg_jax.fe_jax.timo_report import sym, full_pcterr, nonzero_terms, compare_terms, print_6x6, LBL

D = os.path.join(CC, "examples", "data")
SHELL = os.path.join(D, "1d_yaml", "st12_shell.yaml")
BENCH = os.path.join(D, "benchmark", "st12_solid_ref.txt")
DIAG_TOL = {"EA": 8.0, "GA2": 20.0, "GA3": 6.0, "GJ": 38.0, "EI2": 26.0, "EI3": 40.0}


def _compute():
    from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
    from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
    S = sym(np.loadtxt(BENCH))
    RM = sym(rm_timoshenko_6x6(SHELL, 0.0, dshift=None, curved=False, shear="mitc", orient=False))
    KL = sym(gradient_junction_kirchhoff(SHELL, frac=0.0, dshift=None)[0])
    return S, RM, KL


@pytest.mark.skipif(not (os.path.exists(SHELL) and os.path.exists(BENCH)),
                    reason="station-12 data not present")
def test_st12_rm_kl():
    S, RM, KL = _compute()
    eRM, eKL = full_pcterr(RM, S), full_pcterr(KL, S)
    # diagonal vs the (coarse) 2-D solid -- documented per-term bounds
    for i, lab in enumerate(LBL):
        assert RM[i, i] > 0 and KL[i, i] > 0, "non-positive diagonal %s" % lab
        assert abs(eRM[i, i]) < DIAG_TOL[lab], "RM %s off %.2f%%" % (lab, eRM[i, i])
        assert abs(eKL[i, i]) < DIAG_TOL[lab], "KL %s off %.2f%%" % (lab, eKL[i, i])
    # RM and KL agree on every non-zero Cij term (full-matrix shell self-consistency)
    for i, j, tag in nonzero_terms(S, 200.0):
        assert abs(RM[i, j] - KL[i, j]) / max(abs(KL[i, j]), 1e-30) < 0.03, "RM/KL disagree on %s" % tag
    # RM no worse than KL on the transverse-shear terms
    for i in (1, 2):
        assert abs(eRM[i, i]) <= abs(eKL[i, i]) + 1.0, "RM worse than KL on %s" % LBL[i]


def main():
    S, RM, KL = _compute()
    print_6x6(RM, "RM (1-D shell)"); print()
    print_6x6(KL, "KL (1-D shell)")
    print("\nStation-12 (thick web, coarse solid) -- RM and KL vs 2-D solid, every non-zero Cij term:")
    compare_terms(S, {"RM": RM, "KL": KL})


if __name__ == "__main__":
    main()
