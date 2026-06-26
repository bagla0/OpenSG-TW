"""JAX-Kirchhoff with the THEORETICALLY-CORRECT junction treatment.

The Hermite slope DOF is dw/ds = grad(w) . t_e (the warping gradient along THIS
element's tangent).  At a smooth node every incident segment shares the tangent,
so sharing one scalar dw/ds is exact.  At a junction (>=2 distinct tangent
directions: a web meeting the skin, or a kinked corner) sharing one scalar
over-constrains -> grad(w).t_a = grad(w).t_b for non-collinear t -> corrupts the
V1 shear warping -> GA blows up.  Fully decoupling under-constrains -> GA under.

Correct condition (Deo-Yu per-segment transform; equivalent to the FEniCS
C0+interior-penalty C1 in the converged limit): continuity of the full in-plane
gradient g=grad(w) at the node, with each segment's slope reconstructed as
dw/ds_e = g . t_e.  At a junction the >=2 distinct tangents fix g (no null mode);
at a smooth degree-2 node only the tangential component is determined, so the
scalar slope is retained there.

Implementation: assemble in a BROKEN basis (values shared per node = C0; slopes
independent per element-corner), then map to the global gradient-shared basis
with a sparse transform T (broken x global): A_global = T^T A_broken T.  Junction
nodes carry 6 gradient DOFs (gx,gy per component), smooth nodes 3 scalar slopes.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pypardiso
from .msg_mesh import load_yaml, read_mesh, mesh_curvature, offset_oml_to_iml, element_e3_from_yaml
from .msg_materials import compute_ABD_matrix, shift_abd_reference
from .msg_solver import (gauss_legendre_01, compute_element_geometry,
    solve_fluctuation_field, prepare_v1_rhs, finalize_v1_and_compute_deff)
from .msg_hermite import assemble_system_matrices_hermite, build_constraints_hermite
from .orient_plot import auto_emit                    # COMPULSORY e1/e2/e3 orientation plot


def _node_info(red_cells, n_unique, xd2, xd3, tol_deg=30.0):
    """Per node: junction flag (>=2 distinct tangent lines) and a reference tangent."""
    E = len(red_cells); ct = np.cos(np.deg2rad(tol_deg))
    inc = [[] for _ in range(n_unique)]
    for e in range(E):
        for loc in range(2):
            inc[red_cells[e, loc]].append((e, loc))
    is_j = np.zeros(n_unique, bool); t_ref = np.zeros((n_unique, 2))
    for nd in range(n_unique):
        reps = []
        for (e, loc) in inc[nd]:
            t = np.array([xd2[e], xd3[e]]); t = t / (np.linalg.norm(t) + 1e-30)
            for r in reps:
                if abs(np.dot(t, r)) > ct:
                    break
            else:
                reps.append(t)
        is_j[nd] = len(reps) >= 2
        if inc[nd]:
            e0 = inc[nd][0][0]; t0 = np.array([xd2[e0], xd3[e0]])
            t_ref[nd] = t0 / (np.linalg.norm(t0) + 1e-30)
    return is_j, t_ref


def gradient_junction_kirchhoff(yaml_path, frac=0.0, tol_deg=30.0, dshift=None, orient=True, solid_yaml=None):
    if orient:                                  # COMPULSORY orientation plot (once per mesh per process)
        auto_emit(yaml_path, solid_yaml)
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)

    def _abd(i):
        a = compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
        sh = dshift if dshift is not None else (frac * float(sum(i["thick"])) if frac else 0.0)
        return shift_abd_reference(a, sh) if sh else a
    ABD = {ln: _abd(i) for ln, i in layup_db.items()}
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    # reference shift is ABD-only (shift_abd_reference in _abd); nodes are never moved
    # (offset_oml_to_iml removed). Build the mesh on the desired reference surface.
    k22 = jnp.array(mesh_curvature(nodes, cells, elements, is_closed=False))
    ABD_elems = jnp.stack([jnp.array(ABD[ln], dtype=jnp.float64) for ln in lpe])
    hcells = cells[:, [0, -1]]; used = np.unique(hcells)
    f2r = np.full(nodes.shape[0], -1, np.int64); f2r[used] = np.arange(len(used))
    red_cells = f2r[hcells]; corners = nodes[used]; n_unique = len(used)
    L_e, xd2, xd3 = compute_element_geometry(corners, red_cells)
    xd2 = np.asarray(xd2); xd3 = np.asarray(xd3); E = len(red_cells)

    # ---- broken basis: values shared (3*nd+comp); slopes unique per (e,loc,comp) ----
    nb_val = 3 * n_unique
    dof_b = np.zeros((E, 12), np.int64); sidx = {}; sc = 0
    for e in range(E):
        for loc in range(2):
            nd = red_cells[e, loc]
            for comp in range(3):
                dof_b[e, 6 * loc + 2 * comp] = 3 * nd + comp
                dof_b[e, 6 * loc + 2 * comp + 1] = nb_val + sc
                sidx[(e, loc, comp)] = nb_val + sc; sc += 1
    nb = nb_val + sc

    xi_q, W_q = gauss_legendre_01(4)
    Dhh_b, Dhe_b, Dee, Dll_b, Dhl_b, Dle_b = assemble_system_matrices_hermite(
        corners, red_cells, red_cells, ABD_elems, k22, jnp.array(L_e),
        jnp.array(xd2), jnp.array(xd3), xi_q, W_q, nb, dof_map=dof_b)
    C_b, Psi_b = build_constraints_hermite(
        corners, red_cells, red_cells, jnp.array(L_e), jnp.array(xd2), jnp.array(xd3),
        xi_q, W_q, nb, n_unique, dof_map=dof_b)

    # ---- global gradient-shared numbering: junction -> 6 (gx,gy x3), smooth -> 3 ----
    is_j, t_ref = _node_info(red_cells, n_unique, xd2, xd3, tol_deg)
    g_start = np.zeros(n_unique, np.int64); off = 0
    for nd in range(n_unique):
        g_start[nd] = nb_val + off; off += 6 if is_j[nd] else 3
    ng = nb_val + off

    # ---- transform T (nb x ng) ----
    ti, tj, tv = [], [], []
    for i in range(nb_val):                                   # values: identity
        ti.append(i); tj.append(i); tv.append(1.0)
    for (e, loc, comp), sb in sidx.items():                   # slopes
        nd = red_cells[e, loc]
        if is_j[nd]:                                          # dw/ds_e = xd2*gx + xd3*gy
            ti += [sb, sb]
            tj += [int(g_start[nd] + 2 * comp), int(g_start[nd] + 2 * comp + 1)]
            tv += [float(xd2[e]), float(xd3[e])]
        else:                                                 # scalar tangential slope
            s = 1.0 if np.dot([xd2[e], xd3[e]], t_ref[nd]) >= 0 else -1.0
            ti.append(sb); tj.append(int(g_start[nd] + comp)); tv.append(s)
    T = sp.coo_matrix((tv, (ti, tj)), shape=(nb, ng)).tocsr()

    # ---- transform every operator into the global basis ----
    def coo2sp(c):
        return sp.coo_matrix((np.array(c.data), (np.array(c.row), np.array(c.col))),
                             shape=c.shape).tocsr()
    Dhh = coo2sp(Dhh_b); Dll = coo2sp(Dll_b); Dhl = coo2sp(Dhl_b)
    Dhe = np.asarray(Dhe_b.todense()); Dle = np.asarray(Dle_b.todense())
    Cb = np.asarray(C_b)
    Dhh_g = (T.T @ Dhh @ T); Dll_g = (T.T @ Dll @ T); Dhl_g = (T.T @ Dhl @ T)
    Dhe_g = np.asarray(T.T @ Dhe); Dle_g = np.asarray(T.T @ Dle)
    C_g = np.asarray((T.T @ Cb.T).T)                          # 4 x ng  (functional: C_g = C_b @ T)

    # Psi (rigid-body kernel) lives in the PRIMAL space, so it must be built
    # DIRECTLY in the global basis (NOT T^T Psi_b).  3 translations + twist
    # (w2=-y3, w3=y2; exact twist gradient grad(w2)=(0,-1), grad(w3)=(1,0)).
    Psi_g = np.zeros((ng, 4))
    for nd in range(n_unique):
        y2, y3 = corners[nd, 0], corners[nd, 1]
        Psi_g[3 * nd + 0, 0] = 1.0
        Psi_g[3 * nd + 1, 1] = 1.0
        Psi_g[3 * nd + 2, 2] = 1.0
        Psi_g[3 * nd + 1, 3] = -y3
        Psi_g[3 * nd + 2, 3] = y2
        if is_j[nd]:
            Psi_g[g_start[nd] + 2 * 1 + 1, 3] = -1.0          # gy2 = -1
            Psi_g[g_start[nd] + 2 * 2 + 0, 3] = 1.0           # gx3 = +1
        else:
            tx, ty = t_ref[nd]
            Psi_g[g_start[nd] + 1, 3] = -ty                   # dw2/ds = (0,-1).t
            Psi_g[g_start[nd] + 2, 3] = tx                    # dw3/ds = (1,0).t

    # ---- same MSG solve flow as solve_tw_from_yaml, in the global basis ----
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


if __name__ == "__main__":
    from .msg_hermite import timoshenko_from_yaml
    from junction_kirchhoff import junction_kirchhoff
    LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
    D9 = os.path.join(CC, "mh104_9cells", "data")
    cases = [
        ("9-web F=0.3", os.path.join(D9, "shell_9webs_f030.yaml"),
         dict(EA=9.3769e8, GA2=5.3348e7, GA3=5.4219e7, GJ=4.6284e6, EI2=5.7157e6, EI3=2.7212e8)),
        ("9-web F=0.6", os.path.join(D9, "shell_9webs_f060.yaml"),
         dict(GA3=1.070e8)),
        ("mh104 f=0.2", os.path.join(CC, "mh104_thickness_study", "debug", "shell_ref_f020_connect.yaml"),
         dict(GA3=1.3259e7)),
    ]
    for nm, y, sol in cases:
        _, ko, _ = timoshenko_from_yaml(y, frac=0.0); ko = 0.5 * (np.asarray(ko) + np.asarray(ko).T)
        kd, nj = junction_kirchhoff(y, frac=0.0)
        kg, njg, ng = gradient_junction_kirchhoff(y, frac=0.0)
        s3 = sol["GA3"]
        print("\n=== %s  (%d junctions, ng=%d;  solid GA3=%.4e) ===" % (nm, njg, ng, s3))
        print("  GA3: orig %.4e(%+.0f%%)  decouple %.4e(%+.0f%%)  GRADIENT %.4e(%+.0f%%)"
              % (ko[2, 2], 100 * (ko[2, 2] - s3) / s3, kd[2, 2], 100 * (kd[2, 2] - s3) / s3,
                 kg[2, 2], 100 * (kg[2, 2] - s3) / s3))
        if "EA" in sol:
            for k in ["EA", "GA2", "GJ", "EI2", "EI3"]:
                i = LBL.index(k); sv = sol[k]
                print("  %-4s: orig %+.1f%%  decouple %+.1f%%  GRADIENT %+.1f%%  (solid %.3e)"
                      % (k, 100 * (ko[i, i] - sv) / sv, 100 * (kd[i, i] - sv) / sv,
                         100 * (kg[i, i] - sv) / sv, sv))
