"""
Analytical validation: isotropic circular (pipe) cross-section.

The mesh is generated *inside this test* — a closed circle of corner nodes with
midside nodes placed ON the circle, i.e. genuine 3-node CURVED elements, so
``compute_curvature`` recovers k22 = -1/R from each element's own geometry.

Benchmarks against closed-form thin-walled-pipe stiffness
(R = 1 m, h = 0.1 m, E = 1e7 Pa, nu = 0.3).
"""
import numpy as np
import jax.numpy as jnp
import pytest
import pypardiso

from fe_jax import (
    gauss_legendre_01,
    compute_element_geometry,
    compute_curvature,
    build_periodic_dof_map,
    compress_dof_map,
    assemble_system_matrices,
    build_lagrange_constraints,
    build_psi_matrix,
    solve_fluctuation_field,
    prepare_v1_rhs,
    finalize_v1_and_compute_deff,
)


# --------------------------------------------------------------- mesh (in-code)

def make_circular_mesh(R, n_elem):
    """Closed circular cross-section as 3-node CURVED elements.

    Corner and midside nodes both lie on the circle of radius R, so each element
    is a true circular arc (not a straight chord) and carries k22 = -1/R.

    Returns
    -------
    nodes : (2*n_elem+1, 2)  [y2, y3], even rows = corners, odd rows = midsides
    cells : (n_elem, 3)      [corner0, midside, corner1]
    """
    th_c = np.linspace(0.0, 2.0 * np.pi, n_elem + 1)
    th_m = 0.5 * (th_c[:-1] + th_c[1:])
    nodes = np.zeros((2 * n_elem + 1, 2))
    nodes[0::2, 0] = R * np.cos(th_c)
    nodes[0::2, 1] = R * np.sin(th_c)
    nodes[1::2, 0] = R * np.cos(th_m)
    nodes[1::2, 1] = R * np.sin(th_m)
    cells = np.column_stack([
        np.arange(0, 2 * n_elem, 2),
        np.arange(1, 2 * n_elem + 1, 2),
        np.arange(2, 2 * n_elem + 2, 2),
    ]).astype(np.int64)
    return nodes, cells


def abd_isotropic(E, nu, h):
    """CLT ABD (6x6) for a single isotropic layer of thickness h."""
    Q11 = E / (1.0 - nu**2)
    Q12 = nu * Q11
    Q66 = E / (2.0 * (1.0 + nu))
    Q = np.array([[Q11, Q12, 0.0], [Q12, Q11, 0.0], [0.0, 0.0, Q66]])
    Z = np.zeros((3, 3))
    return np.block([[Q * h, Z], [Z, Q * h**3 / 12.0]])


# --------------------------------------------------------------------- fixture

@pytest.fixture(scope="module")
def pipe():
    R, h, E, nu, n_elem = 1.0, 0.1, 1.0e7, 0.3, 40

    nodes, cells = make_circular_mesh(R, n_elem)
    L_e, xd2, xd3 = compute_element_geometry(nodes, cells)
    k22 = np.array(compute_curvature(nodes, cells, is_closed=True))

    ABD = jnp.array(abd_isotropic(E, nu, h))
    ABD_elems = jnp.tile(ABD[None], (n_elem, 1, 1))

    dof_map, n_unique = build_periodic_dof_map(len(nodes), cells, is_closed=True)
    red_cells, _, n_primal = compress_dof_map(dof_map, cells)
    xi_q, W_q = gauss_legendre_01(4)

    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices(
        jnp.array(nodes, dtype=jnp.float64), cells, red_cells,
        ABD_elems, jnp.array(k22), L_e, xd2, xd3, xi_q, W_q, n_primal)
    C_mat = build_lagrange_constraints(
        jnp.array(nodes, dtype=jnp.float64), cells, red_cells,
        L_e, xi_q, W_q, n_primal)
    Psi = build_psi_matrix(
        jnp.array(nodes[:n_unique], dtype=jnp.float64), n_unique, n_primal)
    Dc = C_mat.T

    V0, D1, A_aug = solve_fluctuation_field(Dhh, -np.array(Dhe.todense()), C_mat)
    Ceff = Dee + D1
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n_primal, :]), V0, Ceff,
        V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    C6.block_until_ready()

    G = E / (2.0 * (1.0 + nu))
    Eh, I, xd_sq = E * h, np.pi * R**3, np.pi * R
    II = I + h**2 / 12.0 * xd_sq          # effective second moment
    bend = Eh * II                         # EI
    shear = Eh * II**2 / (2.0 * R**2 * (1.0 + nu) * I)
    analytical = {
        "EA":   Eh * 2.0 * np.pi * R,
        "GA12": shear,
        "GA13": shear,
        "GJ":   2.0 * np.pi * R**3 * G * h,
        "EI2":  bend,
        "EI3":  bend,
    }
    return {"k22": k22, "C6": np.array(C6), "ana": analytical}


# ----------------------------------------------------------------------- tests

def test_curved_elements_recover_k22(pipe):
    """On-circle 3-node elements give k22 = -1/R (here R=1)."""
    assert np.allclose(pipe["k22"], -1.0, atol=1e-6)


@pytest.mark.parametrize("key,idx,tol", [
    ("EA",  0, 0.01),
    ("GA12", 1, 0.05),
    ("GA13", 2, 0.05),
    ("GJ",  3, 0.01),
    ("EI2", 4, 0.01),
    ("EI3", 5, 0.01),
])
def test_against_analytical(pipe, key, idx, tol):
    val = float(pipe["C6"][idx, idx])
    ref = pipe["ana"][key]
    err = abs(val - ref) / ref
    assert err < tol, f"{key}: got {val:.5e}, analytical {ref:.5e}, err {err*100:.2f}%"


def test_symmetry(pipe):
    C = pipe["C6"]
    assert np.linalg.norm(C - C.T) / np.linalg.norm(C) < 1e-6
