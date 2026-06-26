"""Benchmark / regression test: station-12 blade cross-section -- RM & KL shell vs 2-D solid.

Reads the 1-D shell SG YAML (examples/data/1d_yaml/st12_shell.yaml) and computes the Reissner-Mindlin
and Kirchhoff-Love Timoshenko 6x6, benchmarked against the FEniCS 2-D-solid reference
(examples/data/benchmark/st12_solid_ref.txt). This is a thick, web-stiffened airfoil station on a
COARSE solid mesh, so the shell models drift substantially from the filled solid on the bending /
torsion terms (GJ, EI2, EI3 ~20-35 %) -- a documented thin-walled-shell vs 3-D-solid limitation, not a
solver regression. The test therefore (a) holds EA tightly, (b) bounds every term against the solid,
and (c) enforces the RM <= KL transverse-shear property; the wide GJ/EI bounds pin the known gap so a
real regression is still caught.

Run as a test:        pytest tests/test_st12_benchmark.py
Run as a benchmark:   python tests/test_st12_benchmark.py
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

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
D = os.path.join(CC, "examples", "data")
SHELL = os.path.join(D, "1d_yaml", "st12_shell.yaml")
BENCH = os.path.join(D, "benchmark", "st12_solid_ref.txt")
# per-term |%err| bound vs the (coarse) 2-D solid for BOTH RM and KL; GJ/EI2/EI3 pin the thick-section gap
TOL = {"EA": 8.0, "GA2": 20.0, "GA3": 6.0, "GJ": 38.0, "EI2": 26.0, "EI3": 40.0}


def _sym(M):
    M = np.asarray(M, dtype=float); return 0.5 * (M + M.T)


def _pe(C, S, i):
    return 100.0 * (C[i, i] - S[i, i]) / S[i, i]


def _compute():
    from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
    from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
    S = _sym(np.loadtxt(BENCH))
    RM = _sym(rm_timoshenko_6x6(SHELL, 0.0, dshift=None, curved=False, shear="mitc"))
    KL = _sym(gradient_junction_kirchhoff(SHELL, frac=0.0, dshift=None)[0])
    return S, RM, KL


@pytest.mark.skipif(not (os.path.exists(SHELL) and os.path.exists(BENCH)),
                    reason="station-12 data not present")
def test_st12_rm_kl():
    S, RM, KL = _compute()
    for i, lab in enumerate(LBL):
        assert RM[i, i] > 0 and KL[i, i] > 0, "non-positive diagonal %s" % lab
        assert abs(_pe(RM, S, i)) < TOL[lab], "RM %s off %.2f%% (> %.0f)" % (lab, _pe(RM, S, i), TOL[lab])
        assert abs(_pe(KL, S, i)) < TOL[lab], "KL %s off %.2f%% (> %.0f)" % (lab, _pe(KL, S, i), TOL[lab])
    # RM and KL agree closely at a single (non-junction) station
    for i in range(6):
        assert abs(RM[i, i] - KL[i, i]) / abs(KL[i, i]) < 0.03, "RM/KL disagree on %s" % LBL[i]
    # RM no worse than KL on the transverse-shear terms (1% margin, as across the suite)
    for i in (1, 2):
        assert abs(_pe(RM, S, i)) <= abs(_pe(KL, S, i)) + 1.0, "RM worse than KL on %s" % LBL[i]


def main():
    S, RM, KL = _compute()
    print("Station-12 (thick web, coarse solid) -- RM & KL shell vs 2-D solid\n")
    print("  %-5s %14s %14s %14s %10s %10s" % ("term", "RM", "KL", "solid", "RM%err", "KL%err"))
    for i in range(6):
        print("  %-5s %14.5e %14.5e %14.5e %+9.2f %+9.2f" % (LBL[i], RM[i, i], KL[i, i], S[i, i], _pe(RM, S, i), _pe(KL, S, i)))


if __name__ == "__main__":
    main()
