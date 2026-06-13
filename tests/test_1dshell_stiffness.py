"""
Benchmark test: 1Dshell_0.yaml (single glass_triax layup, 52-element airfoil).

Reference values computed with MSG shell quadratic Lagrange (June 2026):
  EA   = 3.265496e+10 N
  GJ   = 4.528373e+10 N·m²
  EI2  = 8.221621e+10 N·m²
  EI3  = 8.210586e+10 N·m²
  GA12 = 5.331989e+09 N
  GA13 = 4.873533e+09 N

Tolerance: 0.5%  (covers mesh/quadrature differences across runs)
"""
import os
import numpy as np
import jax.numpy as jnp
import pytest
import pypardiso

from fe_jax import (
    compute_ABD_matrix,
    load_yaml,
    order_mesh,
    compute_curvature,
    gauss_legendre_01,
    compute_element_geometry,
    build_periodic_dof_map,
    compress_dof_map,
    assemble_system_matrices,
    build_lagrange_constraints,
    build_psi_matrix,
    solve_fluctuation_field,
    prepare_v1_rhs,
    finalize_v1_and_compute_deff,
)


# ---- reference values (benchmark) -----------------------------------------

REF = {
    "EA":   3.265496e+10,
    "GJ":   4.528373e+10,
    "EI2":  8.221621e+10,
    "EI3":  8.210586e+10,
    "GA12": 5.331989e+09,
    "GA13": 4.873533e+09,
}
TOL = 0.005  # 0.5 %


# ---- fixture ---------------------------------------------------------------

@pytest.fixture(scope="module")
def stiffness_1dshell_0(yaml_1dshell_0):
    """Full MSG solve for 1Dshell_0.yaml."""
    nodes_3d, elements, material_db, layup_db, elem_to_layup = load_yaml(yaml_1dshell_0)

    ABD_dict = {}
    for ln, info in layup_db.items():
        ABD_dict[ln], _ = compute_ABD_matrix(
            info['thick'], info['angles'], info['mat_names'], material_db)

    nodes_2d, cells, layup_per_elem, is_closed = order_mesh(
        nodes_3d, elements, elem_to_layup)

    L_e, xd2, xd3 = compute_element_geometry(nodes_2d, cells)
    k22 = jnp.array(compute_curvature(nodes_2d, cells, is_closed))
    ABD_elems = jnp.stack([
        jnp.array(ABD_dict[ln], dtype=jnp.float64) for ln in layup_per_elem])

    n_nodes = len(nodes_2d)
    dof_map, n_unique = build_periodic_dof_map(n_nodes, cells, is_closed)
    red_cells, _, n_primal = compress_dof_map(dof_map, cells)

    xi_q, W_q = gauss_legendre_01(4)

    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices(
        jnp.array(nodes_2d, dtype=jnp.float64),
        cells, red_cells, ABD_elems, k22,
        L_e, xd2, xd3, xi_q, W_q, n_primal)

    C_mat = build_lagrange_constraints(
        jnp.array(nodes_2d, dtype=jnp.float64),
        cells, red_cells, L_e, xi_q, W_q, n_primal)
    Psi = build_psi_matrix(
        jnp.array(nodes_2d[:n_unique], dtype=jnp.float64),
        n_unique, n_primal)
    Dc = C_mat.T

    V0, D1_V0, A_aug = solve_fluctuation_field(
        Dhh, -np.array(Dhe.todense()), C_mat)
    Ceff = Dee + D1_V0

    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)

    R_v1 = np.concatenate(
        [np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)

    Ceff_srt, _, _, _ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n_primal, :]), V0, Ceff,
        V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    Ceff_srt.block_until_ready()

    C4 = np.array(Ceff)
    C6 = np.array(Ceff_srt)
    return {"EB": C4, "Tim": C6}


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
    """6x6 stiffness must be symmetric."""
    C = stiffness_1dshell_0["Tim"]
    assert np.allclose(C, C.T, rtol=1e-5), "Timoshenko stiffness not symmetric"


def test_positive_definite_diagonal(stiffness_1dshell_0):
    """All diagonal entries of the 6x6 stiffness must be positive."""
    C = stiffness_1dshell_0["Tim"]
    diag = np.diag(C)
    assert np.all(diag > 0), f"Non-positive diagonal: {diag}"
