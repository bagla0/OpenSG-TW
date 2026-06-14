"""
Regression test: 1Dshell_0.yaml (single glass_triax layup, 52-element airfoil).

Reference values pin the Hermite C1 MSG-shell TW output:
  EA   = 3.265620e+10 N
  GJ   = 4.447486e+10 N·m²
  EI2  = 7.925902e+10 N·m²
  EI3  = 7.916475e+10 N·m²
  GA12 = 4.583842e+09 N
  GA13 = 4.590813e+09 N

Tolerance: 0.5%  (covers mesh/quadrature differences across runs)
"""
import os
import numpy as np
import pytest

from fe_jax import timoshenko_from_yaml


# ---- reference values (Hermite C1, benchmark) -----------------------------

REF = {
    "EA":   3.265620e+10,
    "GJ":   4.447486e+10,
    "EI2":  7.925902e+10,
    "EI3":  7.916475e+10,
    "GA12": 4.583842e+09,
    "GA13": 4.590813e+09,
}
TOL = 0.005  # 0.5 %

_DATA = os.path.join(os.path.dirname(__file__), "data", "1Dshell_0.yaml")


# ---- fixture ---------------------------------------------------------------

@pytest.fixture(scope="module")
def stiffness_1dshell_0():
    """Hermite C1 MSG-TW solve for 1Dshell_0.yaml."""
    if not os.path.exists(_DATA):
        pytest.skip(f"Test data not found: {_DATA}")
    EB, Tim, _ = timoshenko_from_yaml(_DATA)
    return {"EB": EB, "Tim": Tim}


# ---- tests -----------------------------------------------------------------

@pytest.mark.parametrize("key,row,col", [
    ("EA",  0, 0),
    ("GJ",  1, 1),
    ("EI2", 2, 2),
    ("EI3", 3, 3),
])
def test_eb_diagonal(stiffness_1dshell_0, key, row, col):
    """EB diagonal stiffness within 0.5% of reference."""
    val = float(stiffness_1dshell_0["EB"][row, col])
    ref = REF[key]
    assert abs(val - ref) / ref < TOL, \
        f"{key}: got {val:.6e}, ref {ref:.6e}, diff {abs(val-ref)/ref*100:.3f}%"


@pytest.mark.parametrize("key,idx", [
    ("GA12", 1),
    ("GA13", 2),
])
def test_timoshenko_shear(stiffness_1dshell_0, key, idx):
    """Timoshenko shear stiffness within 0.5% of reference."""
    val = float(stiffness_1dshell_0["Tim"][idx, idx])
    ref = REF[key]
    assert abs(val - ref) / ref < TOL, \
        f"{key}: got {val:.6e}, ref {ref:.6e}, diff {abs(val-ref)/ref*100:.3f}%"


def test_symmetry_timoshenko(stiffness_1dshell_0):
    """6x6 stiffness is symmetric by theory (residual ~1e-3 from the polygonal
    nodal-tangent approximation in the Hermite rigid-twist kernel)."""
    C = stiffness_1dshell_0["Tim"]
    assert np.linalg.norm(C - C.T) / np.linalg.norm(C) < 3e-3, \
        "Timoshenko stiffness not symmetric"


def test_positive_definite_diagonal(stiffness_1dshell_0):
    """All diagonal entries of the 6x6 stiffness must be positive."""
    C = stiffness_1dshell_0["Tim"]
    diag = np.diag(C)
    assert np.all(diag > 0), f"Non-positive diagonal: {diag}"
