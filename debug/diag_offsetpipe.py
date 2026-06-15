"""Decisive reference-consistency test.

A symmetric isotropic pipe whose CENTER is offset to (0, y0) has tension center
(0, y0), so the ext-bend2 coupling must be  C13 = EA * y0  EXACTLY, and this must
be the SAME whether the shell is parametrized at the mid-surface (centroid) or
the outer surface (OML).  If C13/EA != y0, or differs between OML and centroid,
the reference machinery (geometry offset vs ABD parallel-axis shift) is
inconsistent -- the bug behind the airfoil C13 drifting with the reference."""
import os, sys, numpy as np
sys.path.insert(0, r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\opensg_jax")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, pypardiso
from fe_jax import (gauss_legendre_01, compute_element_geometry, compute_curvature,
    make_hermite_mesh, build_hermite_dof_map, compress_hermite_dofs,
    assemble_system_matrices_hermite, build_constraints_hermite,
    solve_fluctuation_field, prepare_v1_rhs, finalize_v1_and_compute_deff)
from fe_jax.msg_materials import shift_abd_reference

R, h, E, nu, n, y0 = 1.0, 0.1, 1.0e7, 0.3, 60, 0.3


def circle(Rad, y0, n):
    th_c = np.linspace(0, 2*np.pi, n+1); th_m = 0.5*(th_c[:-1]+th_c[1:])
    nd = np.zeros((2*n+1, 2))
    nd[0::2, 0] = Rad*np.cos(th_c); nd[0::2, 1] = Rad*np.sin(th_c)+y0
    nd[1::2, 0] = Rad*np.cos(th_m); nd[1::2, 1] = Rad*np.sin(th_m)+y0
    cells = np.column_stack([np.arange(0, 2*n, 2), np.arange(1, 2*n+1, 2),
                             np.arange(2, 2*n+2, 2)]).astype(np.int64)
    return nd, cells


def abd_iso(E, nu, h):
    Q11 = E/(1-nu**2); Q12 = nu*Q11; Q66 = E/(2*(1+nu))
    Q = np.array([[Q11, Q12, 0], [Q12, Q11, 0], [0, 0, Q66]]); Z = np.zeros((3, 3))
    return np.block([[Q*h, Z], [Z, Q*h**3/12]])


def solve(Rad, abd):
    nodes, cells = circle(Rad, y0, n)
    k22 = jnp.array(compute_curvature(nodes, cells, True))
    corners, hcells = make_hermite_mesh(nodes, cells)
    dof_map, n_unique = build_hermite_dof_map(len(corners), is_closed=True)
    red_cells, n_primal = compress_hermite_dofs(dof_map, hcells)
    L_e, xd2, xd3 = compute_element_geometry(corners, hcells)
    ABD = jnp.tile(jnp.array(abd)[None], (n, 1, 1))
    xi, W = gauss_legendre_01(4)
    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices_hermite(
        corners, hcells, red_cells, ABD, k22, L_e, xd2, xd3, xi, W, n_primal)
    Cc, Psi = build_constraints_hermite(
        corners, hcells, red_cells, L_e, xd2, xd3, xi, W, n_primal, n_unique)
    Dc = Cc.T
    V0, D1, A_aug = solve_fluctuation_field(Dhh, -np.array(Dhe.todense()), Cc)
    Ceff = Dee + D1
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)
    Rv1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], 0)
    Vaug = pypardiso.spsolve(A_aug, Rv1)
    C6, *_ = finalize_v1_and_compute_deff(jnp.array(Vaug[:n_primal, :]), V0, Ceff,
                                          V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    C6.block_until_ready()
    return np.array(C6)


print(f"offset pipe: y0 = {y0}  -> tension center y3 must = {y0}")
print(f"{'config':18s}{'xt3 OML':>12s}{'xt3 centroid':>14s}{'centroid err%':>14s}")
for nn, hh in [(60, 0.1), (200, 0.1), (60, 0.02), (200, 0.02)]:
    globals()['n'] = nn; globals()['h'] = hh
    abd_mid = abd_iso(E, nu, hh)
    C_cen = solve(R, abd_mid)
    C_oml = solve(R + hh/2.0, shift_abd_reference(abd_mid, -hh/2.0))
    xc = C_cen[0, 4]/C_cen[0, 0]; xo = C_oml[0, 4]/C_oml[0, 0]
    print(f"  n={nn:3d} h/R={hh:4.2f}  {xo:12.5f}{xc:14.5f}{100*(xc-y0)/y0:14.2f}")
print("  (OML stays exact; if centroid err shrinks with n -> discretization, "
      "if it tracks h/R -> a curvature-coupling formulation error)")
