"""
Analytical validation: isotropic pipe cross-section.

Benchmarks against closed-form EB and Timoshenko stiffness for a thin-walled
circular pipe (R=1 m, h=0.1 m, E=1e7 Pa, nu=0.3).
"""
import numpy as np
import jax.numpy as jnp
import pytest

from fe_jax import (
    gauss_legendre_01,
    make_pipe_mesh,
    compute_element_geometry,
    compute_ABD_isotropic,
    build_periodic_dof_map,
    compress_dof_map,
    assemble_system_matrices,
    build_lagrange_constraints,
    build_psi_matrix,
    solve_fluctuation_field,
    prepare_v1_rhs,
    finalize_v1_and_compute_deff,
)


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def pipe_stiffness():
    """Run full MSG shell Timoshenko solve for an isotropic pipe."""
    R, h, E, nu = 1.0, 0.1, 1e7, 0.3
    n_elem = 40

    nodes, cells, k22 = make_pipe_mesh(R, n_elem)
    L_e, xd2, xd3 = compute_element_geometry(nodes, cells)

    ABD = compute_ABD_isotropic(E, nu, h)
    ABD_elems = jnp.tile(ABD[None], (n_elem, 1, 1))

    dof_map, n_unique = build_periodic_dof_map(len(nodes), cells, is_closed=True)
    red_cells, _, n_primal = compress_dof_map(dof_map, cells)

    xi_q, W_q = gauss_legendre_01(4)

    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices(
        jnp.array(nodes, dtype=jnp.float64),
        cells, red_cells,
        ABD_elems, jnp.array(k22),
        L_e, xd2, xd3, xi_q, W_q, n_primal)

    C_mat = build_lagrange_constraints(
        jnp.array(nodes, dtype=jnp.float64),
        cells, red_cells, L_e, xi_q, W_q, n_primal)
    Psi = build_psi_matrix(
        jnp.array(nodes[:n_unique], dtype=jnp.float64),
        n_unique, n_primal)
    Dc = C_mat.T

    import numpy as _np
    V0, D1_V0, A_aug = solve_fluctuation_field(
        Dhh, -_np.array(Dhe.todense()), C_mat)
    Ceff = Dee + D1_V0

    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)

    import pypardiso
    R_v1 = _np.concatenate(
        [_np.array(bb), _np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)

    Ceff_srt, _, _, _ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n_primal, :]), V0, Ceff,
        V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    Ceff_srt.block_until_ready()

    return {
        "R": R, "h": h, "E": E, "nu": nu,
        "Ceff_EB": np.array(Ceff),
        "Ceff_Tim": np.array(Ceff_srt),
    }


# ------------------------------------------------------------------- tests

def test_eb_ea(pipe_stiffness):
    """EA within 0.2% of Eh * 2πR."""
    d = pipe_stiffness
    analytical = d["E"] * d["h"] * 2 * np.pi * d["R"]
    assert abs(float(d["Ceff_EB"][0, 0]) - analytical) / analytical < 0.002


def test_eb_gj(pipe_stiffness):
    """GJ within 1% of Bredt formula 2πR³ G h."""
    d = pipe_stiffness
    G = d["E"] / (2 * (1 + d["nu"]))
    analytical = 2 * np.pi * d["R"]**3 * G * d["h"]
    assert abs(float(d["Ceff_EB"][1, 1]) - analytical) / analytical < 0.01


def test_eb_ei2(pipe_stiffness):
    """EI2 within 1%."""
    d = pipe_stiffness
    Eh = d["E"] * d["h"]
    I22 = np.pi * d["R"]**3
    xd2_sq = np.pi * d["R"]
    analytical = Eh * (I22 + d["h"]**2 / 12 * xd2_sq)
    assert abs(float(d["Ceff_EB"][2, 2]) - analytical) / analytical < 0.01


def test_eb_ei3(pipe_stiffness):
    """EI3 within 1%."""
    d = pipe_stiffness
    Eh = d["E"] * d["h"]
    I33 = np.pi * d["R"]**3
    xd3_sq = np.pi * d["R"]
    analytical = Eh * (I33 + d["h"]**2 / 12 * xd3_sq)
    assert abs(float(d["Ceff_EB"][3, 3]) - analytical) / analytical < 0.01


def test_timoshenko_ga12(pipe_stiffness):
    """GA12 within 1%."""
    d = pipe_stiffness
    Eh = d["E"] * d["h"]
    I22 = I33 = np.pi * d["R"]**3
    xd2_sq = xd3_sq = np.pi * d["R"]
    analytical = Eh*(I33+d["h"]**2/12*xd3_sq)**2 / (2*d["R"]**2*(1+d["nu"])*I22)
    assert abs(float(d["Ceff_Tim"][1, 1]) - analytical) / analytical < 0.01


def test_timoshenko_ga13(pipe_stiffness):
    """GA13 within 1%."""
    d = pipe_stiffness
    Eh = d["E"] * d["h"]
    I22 = I33 = np.pi * d["R"]**3
    xd2_sq = xd3_sq = np.pi * d["R"]
    analytical = Eh*(I22+d["h"]**2/12*xd2_sq)**2 / (2*d["R"]**2*(1+d["nu"])*I33)
    assert abs(float(d["Ceff_Tim"][2, 2]) - analytical) / analytical < 0.01


def test_symmetry_eb(pipe_stiffness):
    """EB 4x4 stiffness should be symmetric."""
    C = pipe_stiffness["Ceff_EB"]
    assert np.allclose(C, C.T, rtol=1e-6)


def test_symmetry_timoshenko(pipe_stiffness):
    """Timoshenko 6x6 stiffness should be symmetric."""
    C = pipe_stiffness["Ceff_Tim"]
    assert np.allclose(C, C.T, rtol=1e-5)
