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
import jax.numpy as jnp
import pytest
import pypardiso

from fe_jax import (
    compute_ABD_matrix, load_yaml, order_mesh, compute_curvature,
    gauss_legendre_01, compute_element_geometry, build_periodic_dof_map,
    compress_dof_map, assemble_system_matrices, build_lagrange_constraints,
    build_psi_matrix, solve_fluctuation_field, prepare_v1_rhs,
    finalize_v1_and_compute_deff,
)

KEYS = ["EA", "GA12", "GA13", "GJ", "EI2", "EI3"]

# diag of the sorted 6x6 [EA, GA12, GA13, GJ, EI2, EI3]
FENICS_SHELL = np.array([3.2660e10, 4.6665e9, 4.6756e9, 4.5557e10, 7.9279e10, 7.9166e10])
OPENSG_SOLID = np.array([3.2000e10, 4.6860e9, 4.6688e9, 4.5619e10, 7.8053e10, 7.8135e10])


_DATA = os.path.join(os.path.dirname(__file__), "data", "1Dshell_0.yaml")


@pytest.fixture(scope="module")
def jax_6x6():
    if not os.path.exists(_DATA):
        pytest.skip(f"Test data not found: {_DATA}")
    nodes_3d, elements, material_db, layup_db, elem_to_layup = load_yaml(_DATA)
    ABD_dict = {ln: compute_ABD_matrix(i['thick'], i['angles'], i['mat_names'], material_db)[0]
                for ln, i in layup_db.items()}
    nodes_2d, cells, layup_per_elem, is_closed = order_mesh(nodes_3d, elements, elem_to_layup)
    L_e, xd2, xd3 = compute_element_geometry(nodes_2d, cells)
    k22 = jnp.array(compute_curvature(nodes_2d, cells, is_closed))
    ABD_elems = jnp.stack([jnp.array(ABD_dict[ln], dtype=jnp.float64) for ln in layup_per_elem])
    dof_map, n_unique = build_periodic_dof_map(len(nodes_2d), cells, is_closed)
    red_cells, _, n_primal = compress_dof_map(dof_map, cells)
    xi_q, W_q = gauss_legendre_01(4)
    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices(
        jnp.array(nodes_2d, dtype=jnp.float64), cells, red_cells, ABD_elems, k22,
        L_e, xd2, xd3, xi_q, W_q, n_primal)
    C_mat = build_lagrange_constraints(jnp.array(nodes_2d, dtype=jnp.float64),
                                       cells, red_cells, L_e, xi_q, W_q, n_primal)
    Psi = build_psi_matrix(jnp.array(nodes_2d[:n_unique], dtype=jnp.float64), n_unique, n_primal)
    Dc = C_mat.T
    V0, D1, A_aug = solve_fluctuation_field(Dhh, -np.array(Dhe.todense()), C_mat)
    Ceff = Dee + D1
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n_primal, :]), V0, Ceff, V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    C6.block_until_ready()
    return np.diag(np.array(C6))


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
