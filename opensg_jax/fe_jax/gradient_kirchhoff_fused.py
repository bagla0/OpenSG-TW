"""Junction-FUSED gradient-Kirchhoff.

Same result as :func:`gradient_kirchhoff.gradient_junction_kirchhoff`, but the
gradient-continuity tie constraint is folded directly into the local->global
assembly via a *weighted* scatter, instead of assembling the global broken
matrices and then projecting with A_g = T^T A_b T.

This mirrors the periodic DOF-merge (``periodic_dofmap_pardiso.apply_pbc_reduction``),
which is the Boolean special case of the same congruence projection: there each
broken DOF maps to exactly one global DOF with weight 1, so T^T A T degenerates to
an index scatter-add.  At a junction the slope DOF maps to the TWO gradient DOFs
(g_y2, g_y3) with the tangent weights (xd2, xd3), so the scatter carries weights.
No global broken matrices and no explicit T are ever formed.
"""
import os
import numpy as np
import scipy.sparse as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pypardiso
from .msg_mesh import load_yaml, read_mesh, mesh_curvature
from .msg_materials import compute_ABD_matrix, shift_abd_reference
from .msg_solver import (gauss_legendre_01, compute_element_geometry,
    solve_fluctuation_field, prepare_v1_rhs, finalize_v1_and_compute_deff)
from .msg_hermite import assemble_system_matrices_hermite, hermite_shape_functions
from .gradient_kirchhoff import _node_info
from .orient_plot import auto_emit


def gradient_junction_kirchhoff_fused(yaml_path, frac=0.0, tol_deg=30.0,
                                      dshift=None, orient=True, solid_yaml=None):
    if orient:
        auto_emit(yaml_path, solid_yaml)
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)

    def _abd(i):
        a = compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
        sh = dshift if dshift is not None else (frac * float(sum(i["thick"])) if frac else 0.0)
        return shift_abd_reference(a, sh) if sh else a
    ABD = {ln: _abd(i) for ln, i in layup_db.items()}
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    k22 = jnp.array(mesh_curvature(nodes, cells, elements, is_closed=False))
    ABD_elems = jnp.stack([jnp.array(ABD[ln], dtype=jnp.float64) for ln in lpe])
    hcells = cells[:, [0, -1]]; used = np.unique(hcells)
    f2r = np.full(nodes.shape[0], -1, np.int64); f2r[used] = np.arange(len(used))
    red_cells = f2r[hcells]; corners = nodes[used]; n_unique = len(used)
    L_e, xd2, xd3 = compute_element_geometry(corners, red_cells)
    L_e = np.asarray(L_e); xd2 = np.asarray(xd2); xd3 = np.asarray(xd3); E = len(red_cells)

    # ---- reduced gradient-shared numbering: junction -> 6 (gy2,gy3 x3), smooth -> 3 ----
    is_j, t_ref = _node_info(red_cells, n_unique, xd2, xd3, tol_deg)
    nb_val = 3 * n_unique
    g_start = np.zeros(n_unique, np.int64); off = 0
    for nd in range(n_unique):
        g_start[nd] = nb_val + off; off += 6 if is_j[nd] else 3
    ng = nb_val + off

    # ---- per-element local(12) -> reduced map: up to 2 (gcol, weight) per local DOF ----
    # local li = 6*loc + 2*comp (value) or 6*loc + 2*comp + 1 (slope); same order as dof_b
    gcol = np.zeros((E, 12, 2), np.int64)
    gw = np.zeros((E, 12, 2), np.float64)
    for e in range(E):
        for loc in range(2):
            nd = int(red_cells[e, loc])
            for comp in range(3):
                liv = 6 * loc + 2 * comp                          # value DOF
                lis = liv + 1                                     # slope DOF
                gcol[e, liv, 0] = 3 * nd + comp; gw[e, liv, 0] = 1.0   # value: identity
                if is_j[nd]:                                      # slope = xd2*gy2 + xd3*gy3
                    gcol[e, lis, 0] = g_start[nd] + 2 * comp;     gw[e, lis, 0] = xd2[e]
                    gcol[e, lis, 1] = g_start[nd] + 2 * comp + 1; gw[e, lis, 1] = xd3[e]
                else:                                            # scalar tangential slope
                    s = 1.0 if np.dot([xd2[e], xd3[e]], t_ref[nd]) >= 0 else -1.0
                    gcol[e, lis, 0] = g_start[nd] + comp; gw[e, lis, 0] = s

    # ---- per-element broken blocks (12-local) via the existing vmap ----
    xi_q, W_q = gauss_legendre_01(4)
    Dhh_e, Dhe_e, Dee_e, Dll_e, Dhl_e, Dle_e = assemble_system_matrices_hermite(
        corners, red_cells, red_cells, ABD_elems, k22, jnp.array(L_e),
        jnp.array(xd2), jnp.array(xd3), xi_q, W_q, ng, return_blocks=True)
    Dhh_e = np.asarray(Dhh_e); Dhe_e = np.asarray(Dhe_e); Dee_e = np.asarray(Dee_e)
    Dll_e = np.asarray(Dll_e); Dhl_e = np.asarray(Dhl_e); Dle_e = np.asarray(Dle_e)

    # ---- per-element constraint block C_e (4 x 12), same form as build_constraints_hermite ----
    C_e = np.zeros((E, 4, 12))
    for e in range(E):
        val, d1, _ = hermite_shape_functions(xi_q, float(L_e[e]))
        dV = np.asarray(L_e[e] * np.asarray(W_q))
        iv = np.asarray(jnp.einsum("qn,q->n", val, jnp.array(dV)))
        idp = np.asarray(jnp.einsum("qn,q->n", d1, jnp.array(dV)))
        for loc in range(2):
            b = 6 * loc
            v0, v1, p0, p1 = iv[2 * loc], iv[2 * loc + 1], idp[2 * loc], idp[2 * loc + 1]
            C_e[e, 0, b + 0] = v0; C_e[e, 0, b + 1] = v1                  # <w1>
            C_e[e, 1, b + 2] = v0; C_e[e, 1, b + 3] = v1                  # <w2>
            C_e[e, 2, b + 4] = v0; C_e[e, 2, b + 5] = v1                  # <w3>
            C_e[e, 3, b + 2] += xd3[e] * p0; C_e[e, 3, b + 3] += xd3[e] * p1
            C_e[e, 3, b + 4] += -xd2[e] * p0; C_e[e, 3, b + 5] += -xd2[e] * p1

    # ---- weighted scatter (the fused T^T A T, done per element, no global T) ----
    def scatter_sq(Be):                                          # (E,12,12) -> (ng,ng) csr
        rows = np.broadcast_to(gcol[:, :, None, :, None], (E, 12, 12, 2, 2)).ravel()
        cols = np.broadcast_to(gcol[:, None, :, None, :], (E, 12, 12, 2, 2)).ravel()
        ww = gw[:, :, None, :, None] * gw[:, None, :, None, :]
        val = (ww * Be[:, :, :, None, None]).ravel()
        return sp.coo_matrix((val, (rows, cols)), shape=(ng, ng)).tocsr()

    def scatter_right(Be, nc):                                   # (E,12,nc) -> dense (ng,nc)
        rows = np.broadcast_to(gcol[:, :, :, None], (E, 12, 2, nc)).ravel()
        cols = np.broadcast_to(np.arange(nc)[None, None, None, :], (E, 12, 2, nc)).ravel()
        val = (gw[:, :, :, None] * Be[:, :, None, :]).ravel()
        return np.asarray(sp.coo_matrix((val, (rows, cols)), shape=(ng, nc)).todense())

    def scatter_left(Ce, nr):                                    # (E,nr,12) -> dense (nr,ng)
        cols = np.broadcast_to(gcol[:, None, :, :], (E, nr, 12, 2)).ravel()
        rows = np.broadcast_to(np.arange(nr)[None, :, None, None], (E, nr, 12, 2)).ravel()
        val = (Ce[:, :, :, None] * gw[:, None, :, :]).ravel()
        return np.asarray(sp.coo_matrix((val, (rows, cols)), shape=(nr, ng)).todense())

    Dhh_g = scatter_sq(Dhh_e); Dll_g = scatter_sq(Dll_e); Dhl_g = scatter_sq(Dhl_e)
    Dhe_g = scatter_right(Dhe_e, 4); Dle_g = scatter_right(Dle_e, 4)
    Dee = np.asarray(Dee_e).sum(axis=0)
    C_g = scatter_left(C_e, 4)

    # ---- rigid-body kernel built directly in the reduced basis (identical to the T path) ----
    Psi_g = np.zeros((ng, 4))
    for nd in range(n_unique):
        y2, y3 = corners[nd, 0], corners[nd, 1]
        Psi_g[3 * nd + 0, 0] = 1.0; Psi_g[3 * nd + 1, 1] = 1.0; Psi_g[3 * nd + 2, 2] = 1.0
        Psi_g[3 * nd + 1, 3] = -y3; Psi_g[3 * nd + 2, 3] = y2
        if is_j[nd]:
            Psi_g[g_start[nd] + 2 * 1 + 1, 3] = -1.0
            Psi_g[g_start[nd] + 2 * 2 + 0, 3] = 1.0
        else:
            tx, ty = t_ref[nd]
            Psi_g[g_start[nd] + 1, 3] = -ty
            Psi_g[g_start[nd] + 2, 3] = tx

    # ---- identical MSG solve flow, on the reduced space ----
    V0, D1, A_aug = solve_fluctuation_field(Dhh_g, -Dhe_g, jnp.array(C_g))
    Ceff = Dee + D1
    Dc = jnp.array(C_g).T
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, jnp.array(Dhl_g.toarray()), jnp.array(Dll_g.toarray()),
        jnp.array(Dle_g), jnp.array(Psi_g), Dc)
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:ng]), V0, Ceff, V0DllV0, DhlV0, DhlTV0Dle, jnp.array(Psi_g), Dc)
    return np.asarray(0.5 * (C6 + C6.T)), int(is_j.sum()), ng
