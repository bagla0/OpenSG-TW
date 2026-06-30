"""Self-contained [45/-45] anisotropic tube homogenization.

Generates SEPARATE 1D-shell meshes at the OML / center / IML radii -- the nodes
are placed DIRECTLY on each circle (R_out, R_mid, R_in), NOT folded inward from
the OML by ``offset_oml_to_iml`` (the airfoil node-shift that caused junction
folding in mh104).  The plate ABD is analytically reference-shifted
(``shift_abd_reference``) to each surface so the SAME physical wall [R_in,R_out]
is represented with the reference plane at OML / mid / IML.

Two JAX shell models per reference:
  * JAX-Kirchhoff : C1 Hermite (solve_tw assembly, replicated so we can inject
                    explicit nodes + shifted ABD + exact curvature).
  * JAX-RM        : Reissner-Mindlin (msg_rm_timo.timoshenko_rm).

k22 = -1/R_ref  (EXACT hoop curvature -- the tube is a smooth known circle; this
matches the prior bench_tube study.  Flat airfoil walls instead use k22=0 +
refinement; see ref_xml_to_timo_4way_workflow).

Timoshenko order: [EA, GA2, GA3, GJ, EI2, EI3].
"""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "rm"))
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pypardiso

from fe_jax import load_yaml, compute_ABD_matrix
from fe_jax.msg_mesh import read_mesh
from fe_jax.msg_materials import shift_abd_reference
from fe_jax.msg_solver import (gauss_legendre_01, compute_element_geometry,
    solve_fluctuation_field, prepare_v1_rhs, finalize_v1_and_compute_deff)
from fe_jax.msg_hermite import (assemble_system_matrices_hermite,
    build_constraints_hermite)
from fe_jax.msg_rm_timo import timoshenko_rm
from fe_jax.transverse_shear import transverse_shear_stiffness

from gen_meshes import ANI, R_OUT, H, LAYUP, N, REFS, gen_tube_yaml, gen_all  # noqa: F401


def _kirchhoff(nodes2d, cells, lpe, D_by, k22):
    """C1 Hermite Timoshenko 6x6 -- solve_tw_from_yaml assembly with EXPLICIT
    nodes / shifted ABD / curvature injected (verbatim algebra from msg_hermite
    lines 327-353)."""
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


def _flipB(a):
    """Reverse the through-thickness e3 (ply-normal) direction of a 6x6 plate ABD:
    A and D (even in z) are unchanged, the B coupling (odd in z) flips sign.
    Use to express the shell ABD in the VABS/solid e3-OUTWARD convention instead
    of the shell-native OML->IML inward one, so the force-moment beam couplings
    (C14/C25/C36) carry the same sign as the 2D-solid."""
    a = np.array(a, dtype=float).copy()
    a[0:3, 3:6] *= -1.0
    a[3:6, 0:3] *= -1.0
    return a


def homog(yaml_path, R_ref, d_shift, k22_mode="exact", e3="inward", shear="mitc_both"):
    """Return (RM 6x6, Kirchhoff 6x6) for the tube mesh in ``yaml_path`` with the
    reference plane shifted by ``d_shift`` inward and nodes already on R_ref.
    k22_mode: 'exact' -> -1/R_ref hoop curvature; 'zero' -> 0 (faceted).
    e3: 'inward' (shell-native OML->IML) or 'outward' (VABS/solid convention --
    flips the ABD B-block so the coupling signs match the 2D-solid)."""
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    nodes2d = nodes[:, :2]
    elems = cells[:, [0, 1]]
    ne = len(elems)
    # hoop-curvature sign is tied to the traversal: k22=-1/R for a CCW loop
    # (signed area > 0), +1/R for CW.  Computed from the geometry so the curvature
    # stays consistent with the element tangent regardless of node ordering.
    xy = nodes2d[elems[:, 0]]
    area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    ksign = -1.0 if area > 0 else 1.0
    k22 = (ksign / R_ref) * np.ones(ne) if k22_mode == "exact" else np.zeros(ne)

    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        a = shift_abd_reference(a, d_shift) if d_shift else a
        return _flipB(a) if e3 == "outward" else a

    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1, shear=shear)
    KF = _kirchhoff(nodes2d, elems, lpe, D_by, k22)
    return np.asarray(RM), np.asarray(KF)
