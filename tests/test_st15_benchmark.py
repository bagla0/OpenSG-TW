"""Benchmark / regression test: BAR-URC station-15 (thick-web blade station) -- RM & KL shell vs VABS.

Reads the 1-D shell SG YAML (examples/data/1d_yaml/st15_shell.yaml) and computes the Reissner-Mindlin
and Kirchhoff-Love Timoshenko 6x6, benchmarked against the VABS .K reference
(examples/data/benchmark/st15_vabs.K). At this single-cell station RM and KL are close (there is no
multi-cell junction to expose the RM transverse-shear payoff); the shells track VABS to a few percent
except the edgewise bending EI3 (~+14 %, the known thick-web shell-vs-3D drift).

Run as a test:        pytest tests/test_st15_benchmark.py
Run as a benchmark:   python tests/test_st15_benchmark.py
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
SHELL = os.path.join(D, "1d_yaml", "st15_shell.yaml")
KFILE = os.path.join(D, "benchmark", "st15_vabs.K")
# per-term |%err| bound vs VABS for BOTH RM and KL; EI3 carries the thick-web shell drift
TOL = {"EA": 4.0, "GA2": 7.0, "GA3": 14.0, "GJ": 8.0, "EI2": 3.0, "EI3": 18.0}


def _sym(M):
    M = np.asarray(M, dtype=float); return 0.5 * (M + M.T)


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


def _pe(C, S, i):
    return 100.0 * (C[i, i] - S[i, i]) / S[i, i]


def _compute():
    from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
    from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
    S = _sym(_parse_vabs(KFILE))
    RM = _sym(rm_timoshenko_6x6(SHELL, 0.0, dshift=None, curved=False, shear="mitc"))
    KL = _sym(gradient_junction_kirchhoff(SHELL, frac=0.0, dshift=None)[0])
    return S, RM, KL


@pytest.mark.skipif(not (os.path.exists(SHELL) and os.path.exists(KFILE)),
                    reason="station-15 data not present")
def test_st15_rm_kl():
    S, RM, KL = _compute()
    for i, lab in enumerate(LBL):
        assert RM[i, i] > 0 and KL[i, i] > 0, "non-positive diagonal %s" % lab
        assert abs(_pe(RM, S, i)) < TOL[lab], "RM %s off %.2f%% (> %.0f)" % (lab, _pe(RM, S, i), TOL[lab])
        assert abs(_pe(KL, S, i)) < TOL[lab], "KL %s off %.2f%% (> %.0f)" % (lab, _pe(KL, S, i), TOL[lab])
    # RM no worse than KL on the transverse-shear terms (1% margin, as across the suite)
    for i in (1, 2):
        assert abs(_pe(RM, S, i)) <= abs(_pe(KL, S, i)) + 1.0, "RM worse than KL on %s" % LBL[i]


def main():
    S, RM, KL = _compute()
    print("BAR-URC station-15 (thick web) -- RM & KL shell vs VABS .K\n")
    print("  %-5s %14s %14s %14s %10s %10s" % ("term", "RM", "KL", "VABS", "RM%err", "KL%err"))
    for i in range(6):
        print("  %-5s %14.5e %14.5e %14.5e %+9.2f %+9.2f" % (LBL[i], RM[i, i], KL[i, i], S[i, i], _pe(RM, S, i), _pe(KL, S, i)))


if __name__ == "__main__":
    main()
