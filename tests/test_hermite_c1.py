"""
Hermite C1 validation on the isotropic circular (pipe) cross-section.

Mesh generated in-code: a closed circle with corner + midside nodes on the
circle (3-node geometry for k22 = -1/R); the displacement uses Hermite C1 cubic
elements (value + arc-slope DOFs at the corner nodes).  Validated against the
closed-form thin-walled-pipe stiffness.  The Hermite shears are markedly more
accurate than the quadratic-C0 element (≈0.3% vs ≈3% for GA12).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
import numpy as np
import jax.numpy as jnp
import pytest
import pypardiso

from fe_jax import (
    gauss_legendre_01, compute_element_geometry, compute_curvature,
    make_hermite_mesh, build_hermite_dof_map, compress_hermite_dofs,
    assemble_system_matrices_hermite, build_constraints_hermite,
    solve_fluctuation_field, prepare_v1_rhs, finalize_v1_and_compute_deff,
)

KEYS = ["EA", "GA12", "GA13", "GJ", "EI2", "EI3"]


def _circle_3node(R, n):
    th_c = np.linspace(0.0, 2*np.pi, n + 1)
    th_m = 0.5 * (th_c[:-1] + th_c[1:])
    nodes = np.zeros((2*n + 1, 2))
    nodes[0::2, 0] = R*np.cos(th_c); nodes[0::2, 1] = R*np.sin(th_c)
    nodes[1::2, 0] = R*np.cos(th_m); nodes[1::2, 1] = R*np.sin(th_m)
    cells = np.column_stack([np.arange(0, 2*n, 2), np.arange(1, 2*n+1, 2),
                             np.arange(2, 2*n+2, 2)]).astype(np.int64)
    return nodes, cells


def _abd_iso(E, nu, h):
    Q11 = E/(1-nu**2); Q12 = nu*Q11; Q66 = E/(2*(1+nu))
    Q = np.array([[Q11, Q12, 0], [Q12, Q11, 0], [0, 0, Q66]]); Z = np.zeros((3, 3))
    return np.block([[Q*h, Z], [Z, Q*h**3/12]])


@pytest.fixture(scope="module")
def pipe_hermite():
    R, h, E, nu, n = 1.0, 0.1, 1.0e7, 0.3, 40
    nodes, cells = _circle_3node(R, n)
    k22 = jnp.array(compute_curvature(nodes, cells, True))

    corners, hcells = make_hermite_mesh(nodes, cells)
    dof_map, n_unique = build_hermite_dof_map(len(corners), is_closed=True)
    red_cells, n_primal = compress_hermite_dofs(dof_map, hcells)
    L_e, xd2, xd3 = compute_element_geometry(corners, hcells)
    ABD_elems = jnp.tile(jnp.array(_abd_iso(E, nu, h))[None], (n, 1, 1))
    xi, W = gauss_legendre_01(4)

    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices_hermite(
        corners, hcells, red_cells, ABD_elems, k22, L_e, xd2, xd3, xi, W, n_primal)
    C, Psi = build_constraints_hermite(
        corners, hcells, red_cells, L_e, xd2, xd3, xi, W, n_primal, n_unique)
    Dc = C.T
    V0, D1, A_aug = solve_fluctuation_field(Dhh, -np.array(Dhe.todense()), C)
    Ceff = Dee + D1
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n_primal, :]), V0, Ceff, V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    C6.block_until_ready()

    G = E/(2*(1+nu)); Eh, I, xs = E*h, np.pi*R**3, np.pi*R; II = I + h**2/12*xs
    shear = Eh*II**2/(2*R**2*(1+nu)*I)
    ana = {"EA": Eh*2*np.pi*R, "GA12": shear, "GA13": shear,
           "GJ": 2*np.pi*R**3*G*h, "EI2": Eh*II, "EI3": Eh*II}
    return {"k22": np.array(k22), "diag": np.diag(np.array(C6)), "ana": ana, "C6": np.array(C6)}


def test_k22_recovered(pipe_hermite):
    assert np.allclose(pipe_hermite["k22"], -1.0, atol=1e-6)


@pytest.mark.parametrize("key,idx,tol", [
    ("EA", 0, 0.01), ("GA12", 1, 0.01), ("GA13", 2, 0.01),
    ("GJ", 3, 0.01), ("EI2", 4, 0.01), ("EI3", 5, 0.01),
])
def test_against_analytical(pipe_hermite, key, idx, tol):
    val = float(pipe_hermite["diag"][idx]); ref = pipe_hermite["ana"][key]
    err = abs(val - ref) / ref
    assert err < tol, f"{key}: {val:.5e} vs {ref:.5e} = {err*100:.2f}%"


def test_shear_symmetry(pipe_hermite):
    """Circular section -> GA12 == GA13 (Hermite keeps this tight)."""
    d = pipe_hermite["diag"]
    assert abs(d[1] - d[2]) / d[1] < 0.01


def test_symmetry(pipe_hermite):
    # The Timoshenko matrix is symmetric by theory; the small residual (~1e-3)
    # comes from the polygonal nodal-tangent approximation in the rigid-twist
    # kernel Psi (the slope DOF of w=(-y3,y2) at a corner). Diagonals are <1%.
    C = pipe_hermite["C6"]
    assert np.linalg.norm(C - C.T) / np.linalg.norm(C) < 3e-3


if __name__ == "__main__":
    import sys as _s
    R, h, E, nu, n = 1.0, 0.1, 1.0e7, 0.3, 40
    # reuse the fixture body by calling it directly is awkward; just print a note
    print("Run via:  pytest tests/test_hermite_c1.py -s")
