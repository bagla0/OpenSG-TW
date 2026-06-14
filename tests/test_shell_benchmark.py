"""
Shell-YAML benchmark: 1Dshell_0 cross-section.

Solves the MSG-shell Timoshenko 6x6 with the JAX (TW) code and compares the
diagonal against two external references:

  * FEniCS-TW  -- OpenSG shell ``compute_timo_boun`` on the same 1Dshell_0.yaml
  * OpenSG SOLID -- ``compute_timo_boun`` (solid) on 2Dsolid_0.yaml, the same
                    cross-section meshed as a 2D solid (the "exact" benchmark).

Both reference matrices were produced by running OpenSG/FEniCS (dolfinx 0.8.0).
Run with ``pytest -s`` to print the comparison table.
"""
import os
import numpy as np
import pytest

from fe_jax import timoshenko_from_yaml

KEYS = ["EA", "GA12", "GA13", "GJ", "EI2", "EI3"]

# diag of the sorted 6x6 [EA, GA12, GA13, GJ, EI2, EI3]
FENICS_SHELL = np.array([3.2660e10, 4.6665e9, 4.6756e9, 4.5557e10, 7.9279e10, 7.9166e10])
OPENSG_SOLID = np.array([3.2000e10, 4.6860e9, 4.6688e9, 4.5619e10, 7.8053e10, 7.8135e10])


_DATA = os.path.join(os.path.dirname(__file__), "data", "1Dshell_0.yaml")


@pytest.fixture(scope="module")
def jax_6x6():
    if not os.path.exists(_DATA):
        pytest.skip(f"Test data not found: {_DATA}")
    _, Tim, _ = timoshenko_from_yaml(_DATA)
    return np.diag(Tim)


def test_benchmark_table(jax_6x6):
    """Print JAX vs FEniCS-TW vs OpenSG-solid and check agreement."""
    print("\n  1Dshell_0 Timoshenko diagonal: JAX-TW vs FEniCS-TW vs OpenSG-SOLID")
    print("  " + "-" * 78)
    print("  %-5s %14s %14s %14s %8s %8s" %
          ("term", "JAX-TW", "FEniCS-TW", "SOLID", "dF%", "dSol%"))
    print("  " + "-" * 78)
    for i, k in enumerate(KEYS):
        j, f, s = jax_6x6[i], FENICS_SHELL[i], OPENSG_SOLID[i]
        print("  %-5s %14.5e %14.5e %14.5e %7.2f %7.2f" %
              (k, j, f, s, (j - f) / f * 100, (j - s) / s * 100))
    print("  " + "-" * 78)

    # JAX must track the FEniCS shell closely; shears are softer (flat polygon
    # vs FEniCS's frame-smoothed curvature) so allow a wider band there.
    tol_F = {"EA": 0.01, "GJ": 0.03, "EI2": 0.01, "EI3": 0.01, "GA12": 0.05, "GA13": 0.05}
    for i, k in enumerate(KEYS):
        dF = abs(jax_6x6[i] - FENICS_SHELL[i]) / FENICS_SHELL[i]
        assert dF < tol_F[k], f"{k}: JAX {jax_6x6[i]:.5e} vs FEniCS {FENICS_SHELL[i]:.5e} = {dF*100:.2f}%"


@pytest.mark.parametrize("key,idx", [("EA", 0), ("EI2", 4), ("EI3", 5)])
def test_close_to_solid(jax_6x6, key, idx):
    """Axial and bending stiffness within 2.5% of the exact solid benchmark."""
    d = abs(jax_6x6[idx] - OPENSG_SOLID[idx]) / OPENSG_SOLID[idx]
    assert d < 0.025, f"{key}: JAX {jax_6x6[idx]:.5e} vs solid {OPENSG_SOLID[idx]:.5e} = {d*100:.2f}%"
