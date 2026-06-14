"""
Five-cross-section benchmark: JAX-TW vs FEniCS-TW for 1Dshell_0 .. 1Dshell_4.

For each cross-section the MSG-shell Timoshenko 6x6 diagonal is computed with the
JAX (TW) code and compared element-by-element against OpenSG's FEniCS shell
``compute_timo_boun`` (dolfinx 0.8.0).  Run with ``pytest -s`` to print the
percent-difference table.
"""
import os
import sys
# allow `python tests/test_five_sections.py` (conftest only runs under pytest)
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


def _solve_jax(yaml_path):
    """Hermite C1 MSG-TW Timoshenko 6x6 diagonal + mesh-completeness flag."""
    _, Tim, complete = timoshenko_from_yaml(yaml_path)
    return np.diag(Tim), complete


@pytest.fixture(scope="module")
def jax_five():
    out = {}
    for i in range(5):
        path = os.path.join(DATA_DIR, f"1Dshell_{i}.yaml")
        if not os.path.exists(path):
            pytest.skip(f"missing {path}")
        out[i] = _solve_jax(path)   # (diag, complete)
    return out


def test_table(jax_five):
    """Print JAX-TW vs FEniCS-TW % diff for the first five cross-sections.

    Sections flagged ``web`` have a shear web that order_mesh's single-loop
    chaining does not yet capture (open topology limitation, not a TW bug).
    """
    print("\n  JAX-TW vs FEniCS-TW  |  percent difference (100*(JAX-FE)/FE)")
    print("  sec " + " ".join(f"{k:>8s}" for k in KEYS) + "   topology")
    print("  " + "-" * 72)
    for i in range(5):
        diag, complete = jax_five[i]
        fe = FENICS_TW.get(i)
        diffs = [(diag[j] - fe[j]) / fe[j] * 100 for j in range(6)]
        tag = "single-loop" if complete else "WEB (incomplete)"
        print(f"  {i:<3d} " + " ".join(f"{d:>7.2f}%" for d in diffs) + f"   {tag}")
    print("  " + "-" * 72)


def test_single_loop_sections_match_fenics(jax_five):
    """Fully-chained (single-loop) sections must track FEniCS-TW.

    EA/EI within 1%, transverse shears + GJ within 6% (flat polygon vs FEniCS
    frame-smoothed curvature). Webbed sections are reported but not asserted
    until multi-component meshing lands.
    """
    tol = {"EA": 0.01, "GA12": 0.06, "GA13": 0.06, "GJ": 0.06, "EI2": 0.01, "EI3": 0.01}
    checked = 0
    for i in range(5):
        diag, complete = jax_five[i]
        if not complete:
            continue
        checked += 1
        fe = FENICS_TW[i]
        for j, k in enumerate(KEYS):
            d = abs(diag[j] - fe[j]) / fe[j]
            assert d < tol[k], f"section {i} {k}: JAX {diag[j]:.4e} vs FEniCS {fe[j]:.4e} = {d*100:.2f}%"
    assert checked >= 3, f"expected >=3 single-loop sections, got {checked}"


@pytest.mark.parametrize("sec", [i for i in range(5)])
def test_diag_positive(jax_five, sec):
    """6x6 diagonal must be positive for every cross-section."""
    diag, _ = jax_five[sec]
    assert np.all(diag > 0), f"section {sec}: {diag}"


# --------------------------------------------------------------- run directly
# `python tests/test_five_sections.py` actually solves and prints the table
# (the pytest test functions above only execute under pytest).
if __name__ == "__main__":
    print("JAX-TW vs FEniCS-TW  |  percent difference (100*(JAX-FE)/FE)")
    print("sec " + " ".join(f"{k:>8s}" for k in KEYS) + "   topology")
    print("-" * 74)
    for i in range(5):
        path = os.path.join(DATA_DIR, f"1Dshell_{i}.yaml")
        if not os.path.exists(path):
            print(f"{i:<3d}  (missing {path})")
            continue
        diag, complete = _solve_jax(path)
        fe = FENICS_TW[i]
        diffs = [(diag[j] - fe[j]) / fe[j] * 100 for j in range(6)]
        tag = "single-loop" if complete else "WEB (incomplete)"
        print(f"{i:<3d} " + " ".join(f"{d:>7.2f}%" for d in diffs) + f"   {tag}")
    print("-" * 74)
    print("JAX absolute diagonals [EA, GA12, GA13, GJ, EI2, EI3]:")
    for i in range(5):
        path = os.path.join(DATA_DIR, f"1Dshell_{i}.yaml")
        if os.path.exists(path):
            diag, _ = _solve_jax(path)
            print(f"  sec {i}: " + " ".join(f"{v:.4e}" for v in diag))
