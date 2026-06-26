"""Benchmark / regression test: two-cell [-45] anisotropic tube (ASC paper, bagla2025asc).

A multi-cell composite tube with an internal shear web. Benchmarked on the FULL 6x6 (every non-zero
Cij term, not just the diagonal) against the FEniCS 2-D solid:
  1. the JAX 2-D solid reproduces the FEniCS 2-D-solid reference on every non-zero term;
  2. RM is within a few % on EVERY non-zero Cij term (full-6x6 accuracy of the reduced shell);
  3. RM is at least as good as KL on the transverse-shear terms GA2(C22),GA3(C33) -- the web-junction payoff;
  4. KL really does lose the shear (so the benchmark is discriminating).

Run as a test:        pytest tests/test_twocell_m45_benchmark.py
Run as a benchmark:   python tests/test_twocell_m45_benchmark.py
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
from opensg_jax.fe_jax.timo_report import sym, full_pcterr, nonzero_terms, compare_terms, print_6x6, LBL

D = os.path.join(CC, "examples", "data")
SHELL = os.path.join(D, "1d_yaml", "tube2cell_m45_shell.yaml")
SOLIDY = os.path.join(D, "2d_yaml", "tube2cell_m45_solid.yaml")
BENCH = os.path.join(D, "benchmark", "tube2cell_m45_solid_ref.txt")


def _wall_t(meshp):
    d = yaml.safe_load(open(meshp)); return sum(float(p[1]) for p in d["sections"][0]["layup"])


def _compute():
    from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
    from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
    from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
    T = _wall_t(SHELL)
    S = sym(np.loadtxt(BENCH))
    RM = sym(rm_timoshenko_6x6(SHELL, 0.0, dshift=T / 2, curved=True, shear="mitc", orient=False))
    KL = sym(gradient_junction_kirchhoff(SHELL, frac=0.0, dshift=T / 2)[0])
    JX = sym(compute_timo_from_yaml(SOLIDY, verbose=False))
    return S, RM, KL, JX


@pytest.mark.skipif(not (os.path.exists(SHELL) and os.path.exists(BENCH)),
                    reason="two-cell [-45] data not present")
def test_twocell_m45_full6x6():
    S, RM, KL, JX = _compute()
    nz = nonzero_terms(S)                                  # all non-zero Cij (max/1000 cutoff)
    eJX, eRM, eKL = full_pcterr(JX, S), full_pcterr(RM, S), full_pcterr(KL, S)
    # 1. JAX 2-D solid == FEniCS 2-D-solid reference on every non-zero term
    for i, j, tag in nz:
        assert abs(eJX[i, j]) < 1.0, "JAX-solid %s off %.3f%%" % (tag, eJX[i, j])
    # 2. RM within a few % on EVERY non-zero Cij term (full-6x6 accuracy)
    for i, j, tag in nz:
        assert abs(eRM[i, j]) < 4.0, "RM %s off %.2f%%" % (tag, eRM[i, j])
    # 3. RM <= KL on the transverse-shear terms (the multi-cell junction payoff)
    for i in (1, 2):
        assert abs(eRM[i, i]) <= abs(eKL[i, i]) + 1.0, "RM not <= KL on %s" % LBL[i]
    # 4. KL really loses the shear here (discriminating benchmark)
    assert max(abs(eKL[1, 1]), abs(eKL[2, 2])) > 8.0


def main():
    S, RM, KL, JX = _compute()
    print_6x6(S, "FEniCS 2-D solid (reference)"); print()
    print_6x6(RM, "RM (1-D shell)"); print()
    print_6x6(KL, "KL (1-D shell)")
    print("\nRM, KL and the JAX 2-D solid vs the FEniCS 2-D solid -- every non-zero Cij term:")
    compare_terms(S, {"RM": RM, "KL": KL, "JAXsolid": JX})
    print("\n  RM recovers GA2,GA3 (and their couplings C13,C25,C36) across the web where KL drops them.")


if __name__ == "__main__":
    main()
