"""Benchmark / regression test: two-cell [-45] anisotropic tube (ASC paper, bagla2025asc).

A multi-cell composite cross-section with an internal-web junction. Asserts:
  1. the JAX 2-D solid sits on the FEniCS 2-D-solid reference (every diagonal < 0.5%);
  2. RM is at least as good as KL on the transverse-shear terms GA2,GA3 (the RM payoff across the junction);
  3. RM keeps GA2,GA3 within a few % of the solid reference.

Run as a benchmark:   python tests/test_twocell_m45_benchmark.py
Run as a test:        pytest tests/test_twocell_m45_benchmark.py
"""
import os
import sys
import numpy as np
import yaml
import pytest

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
D = os.path.join(CC, "examples", "data")
SHELL = os.path.join(D, "1d_yaml", "tube2cell_m45_shell.yaml")
SOLIDY = os.path.join(D, "2d_yaml", "tube2cell_m45_solid.yaml")
BENCH = os.path.join(D, "benchmark", "tube2cell_m45_solid_ref.txt")


def _sym(M):
    M = np.asarray(M, dtype=float); return 0.5 * (M + M.T)


def _wall_t(meshp):
    d = yaml.safe_load(open(meshp)); return sum(float(p[1]) for p in d["sections"][0]["layup"])


def _compute():
    from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
    from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
    from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
    T = _wall_t(SHELL)
    S = _sym(np.loadtxt(BENCH))
    KL = _sym(gradient_junction_kirchhoff(SHELL, frac=0.0, dshift=T / 2)[0])
    RM = _sym(rm_timoshenko_6x6(SHELL, 0.0, dshift=T / 2, curved=True, shear="mitc"))
    JX = _sym(compute_timo_from_yaml(SOLIDY, verbose=False))
    return S, KL, RM, JX


def _pe(C, S, i):
    return 100.0 * (C[i, i] - S[i, i]) / S[i, i]


@pytest.mark.skipif(not (os.path.exists(SHELL) and os.path.exists(BENCH)),
                    reason="two-cell [-45] data not present")
def test_twocell_m45():
    S, KL, RM, JX = _compute()
    # 1. JAX 2-D solid reproduces the 2-D solid reference
    for i in range(6):
        assert abs(_pe(JX, S, i)) < 0.5, "JAX-solid %s off %.3f%%" % (LBL[i], _pe(JX, S, i))
    # 2. RM <= KL on the transverse-shear terms (the multi-cell junction payoff)
    for i in (1, 2):
        assert abs(_pe(RM, S, i)) <= abs(_pe(KL, S, i)) + 1.0, "RM not <= KL on %s" % LBL[i]
    # 3. RM keeps the shear terms within a few % of the solid
    for i in (1, 2):
        assert abs(_pe(RM, S, i)) < 6.0, "RM %s off %.2f%%" % (LBL[i], _pe(RM, S, i))


def main():
    S, KL, RM, JX = _compute()
    print("Two-cell [-45] anisotropic tube (ASC) -- KL vs RM vs JAX-solid, vs 2-D solid reference\n")
    print("  %-5s %10s %10s %12s" % ("term", "KL%err", "RM%err", "JAXsol%err"))
    for i in range(6):
        print("  %-5s %+9.2f %+9.2f %+11.4f" % (LBL[i], _pe(KL, S, i), _pe(RM, S, i), _pe(JX, S, i)))
    print("\n  RM recovers GA2,GA3 across the internal-web junction where KL drops them.")


if __name__ == "__main__":
    main()
