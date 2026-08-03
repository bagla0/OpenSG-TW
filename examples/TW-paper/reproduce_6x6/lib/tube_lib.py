"""Self-contained smooth-tube homogenization (single-cell circle, EXACT k22).

Two JAX shell models per call:
  * JAX-Kirchhoff : C1 Hermite (gradient-Kirchhoff assembly, explicit nodes +
                    shifted ABD + exact curvature injected).
  * JAX-RM        : Reissner-Mindlin (msg_rm_timo.timoshenko_rm, MITC).

k22 = -1/R_ref  (EXACT hoop curvature -- the tube is a smooth known circle).
Timoshenko order: [EA, GA2, GA3, GJ, EI2, EI3].
"""
import os
import sys

import numpy as np

# --- portable bootstrap: locate the OpenSG-TW repo root (a dir with opensg_jax/fe_jax) ---
_HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(start):
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "opensg_jax", "fe_jax")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("OpenSG-TW repo root (opensg_jax/fe_jax) not found above %s" % start)


_ROOT = _repo_root(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pypardiso

from opensg_jax.fe_jax import load_yaml, compute_ABD_matrix
from opensg_jax.fe_jax.msg_mesh import read_mesh
from opensg_jax.fe_jax.msg_materials import shift_abd_reference
from opensg_jax.fe_jax.msg_solver import (gauss_legendre_01, compute_element_geometry,
    solve_fluctuation_field, prepare_v1_rhs, finalize_v1_and_compute_deff)
from opensg_jax.fe_jax.msg_hermite import (assemble_system_matrices_hermite,
    build_constraints_hermite)
from opensg_jax.fe_jax.msg_rm_timo import timoshenko_rm
from opensg_jax.fe_jax.transverse_shear import transverse_shear_stiffness

from gen_meshes import ANI, R_OUT, H, LAYUP, N, REFS, gen_tube_yaml, gen_all  # noqa: F401


def _kirchhoff(nodes2d, cells, lpe, D_by, k22):
    """C1 Hermite Timoshenko 6x6 with explicit nodes / shifted ABD / curvature injected."""
    hcells = cells[:, [0, -1]]
    used = np.unique(hcells)
    f2r = np.full(nodes2d.shape[0], -1, dtype=np.int64)
    f2r[used] = np.arange(len(used))
    red_cells = f2r[hcells]
    corners = nodes2d[used]
    n_unique = len(used)
    n_primal = 6 * n_unique
    L_e, xd2, xd3 = compute_element_geometry(corners, red_cells)
    xi_q, W_q = gauss_legendre_01(4)
    ABD_elems = jnp.stack([jnp.array(D_by[ln], dtype=jnp.float64) for ln in lpe])
    k22j = jnp.array(k22)
    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices_hermite(
        corners, red_cells, red_cells, ABD_elems, k22j, L_e, xd2, xd3, xi_q, W_q, n_primal)
    C, Psi = build_constraints_hermite(
        corners, red_cells, red_cells, L_e, xd2, xd3, xi_q, W_q, n_primal, n_unique)
    Dc = C.T
    V0, D1, A_aug = solve_fluctuation_field(Dhh, -np.array(Dhe.todense()), C)
    Ceff = Dee + D1
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, _Btim, _Ctim, _V1 = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n_primal, :]), V0, Ceff, V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    C6.block_until_ready()
    return np.asarray(C6)


def homog(yaml_path, R_ref, d_shift, k22_mode="exact", shear="mitc_both"):
    """Return (RM 6x6, Kirchhoff 6x6) for the smooth-tube mesh in `yaml_path`.

    The reference plane is shifted by `d_shift` inward (ABD-only; nodes stay put).
    k22_mode: 'exact' -> -1/R_ref hoop curvature; 'zero' -> 0 (faceted)."""
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    nodes2d = nodes[:, :2]
    elems = cells[:, [0, 1]]
    ne = len(elems)
    # k22 sign is tied to the traversal: -1/R for a CCW loop (signed area > 0).
    xy = nodes2d[elems[:, 0]]
    area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    ksign = -1.0 if area > 0 else 1.0
    k22 = (ksign / R_ref) * np.ones(ne) if k22_mode == "exact" else np.zeros(ne)

    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(a, d_shift) if d_shift else a

    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1, shear=shear)
    KF = _kirchhoff(nodes2d, elems, lpe, D_by, k22)
    return np.asarray(RM), np.asarray(KF)
