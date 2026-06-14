"""
Five-cross-section benchmark: JAX-TW (Hermite C1) vs FEniCS-TW, 1Dshell_0 .. _4.

The JAX mesh is taken straight from the YAML connectivity (no chaining), so
shear-webbed sections (3, 4) are fully represented.  For each cross-section the
MSG-shell Timoshenko 6x6 diagonal is compared against OpenSG's FEniCS shell
``compute_timo_boun`` (dolfinx 0.8.0).  Run with ``pytest -s`` (or
``python tests/test_five_sections.py``) to print the percent-difference table.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
import numpy as np
import pytest

from fe_jax import timoshenko_from_yaml

KEYS = ["EA", "GA12", "GA13", "GJ", "EI2", "EI3"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# FEniCS-TW sorted-6x6 diagonals (OpenSG shell compute_timo_boun, dolfinx 0.8.0)
# rows: 1Dshell_0 .. 1Dshell_4   cols: [EA, GA12, GA13, GJ, EI2, EI3]
FENICS_TW = {
    0: [3.266036e10, 4.666542e9, 4.675592e9, 4.555683e10, 7.927934e10, 7.916552e10],
    1: [2.819401e10, 3.873506e9, 3.378078e9, 3.244199e10, 6.376697e10, 6.125958e10],
    2: [2.245040e10, 3.133497e9, 1.871966e9, 1.572421e10, 3.332955e10, 4.119837e10],
    3: [1.821495e10, 2.410690e9, 1.179624e9, 6.160945e9, 1.511559e10, 2.804726e10],
    4: [1.566355e10, 1.885614e9, 7.678409e8, 3.543155e9, 9.901202e9, 2.190045e10],
}

# EA/EI track FEniCS tightly; transverse shears + GJ are slightly softer
# (flat polygon vs FEniCS frame-smoothed curvature).
TOL = {"EA": 0.01, "GA12": 0.03, "GA13": 0.03, "GJ": 0.03, "EI2": 0.01, "EI3": 0.01}


def _diag(yaml_path):
    _, Tim, _ = timoshenko_from_yaml(yaml_path)
    return np.diag(Tim)


@pytest.fixture(scope="module")
def jax_five():
    out = {}
    for i in range(5):
        path = os.path.join(DATA_DIR, f"1Dshell_{i}.yaml")
        if not os.path.exists(path):
            pytest.skip(f"missing {path}")
        out[i] = _diag(path)
    return out


def _print_table(diags):
    print("\n  JAX-TW (Hermite C1) vs FEniCS-TW  |  100*(JAX-FE)/FE  [%]")
    print("  sec " + " ".join(f"{k:>8s}" for k in KEYS))
    print("  " + "-" * 60)
    for i in range(5):
        fe = FENICS_TW[i]
        diffs = [(diags[i][j] - fe[j]) / fe[j] * 100 for j in range(6)]
        print(f"  {i:<3d} " + " ".join(f"{d:>7.2f}%" for d in diffs))
    print("  " + "-" * 60)


def test_table(jax_five):
    _print_table(jax_five)


@pytest.mark.parametrize("sec", list(range(5)))
def test_matches_fenics(jax_five, sec):
    """Every section (incl. webbed 3, 4) tracks FEniCS-TW within tolerance."""
    fe = FENICS_TW[sec]
    for j, k in enumerate(KEYS):
        d = abs(jax_five[sec][j] - fe[j]) / fe[j]
        assert d < TOL[k], f"section {sec} {k}: JAX {jax_five[sec][j]:.4e} vs FEniCS {fe[j]:.4e} = {d*100:.2f}%"


@pytest.mark.parametrize("sec", list(range(5)))
def test_diag_positive(jax_five, sec):
    assert np.all(jax_five[sec] > 0), f"section {sec}: {jax_five[sec]}"


if __name__ == "__main__":
    diags = {}
    for i in range(5):
        p = os.path.join(DATA_DIR, f"1Dshell_{i}.yaml")
        if os.path.exists(p):
            diags[i] = _diag(p)
    _print_table(diags)
    print("\n  JAX absolute diagonals [EA, GA12, GA13, GJ, EI2, EI3]:")
    for i in diags:
        print(f"    sec {i}: " + " ".join(f"{v:.4e}" for v in diags[i]))
