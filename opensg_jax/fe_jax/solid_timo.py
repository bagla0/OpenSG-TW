"""JAX OpenSG-TW: 2D-solid Timoshenko 6x6 homogenizer driven by a 2D-solid SG YAML.

A JAX analogue of the FEniCS `opensg/core/solid.py::compute_timo_boun`, reusing the MSG assembly/solve
pipeline of `Beam_solid.py` (verbatim functions copied here so Beam_solid.py is left untouched) but reading
the cross-section from a 2D-solid YAML via `segment.read_solid_yaml` instead of a `.sc` file.

    from solid_timo import compute_timo_from_yaml
    C6 = compute_timo_from_yaml("solid_bar_r050.yaml")   # -> 6x6 Timoshenko stiffness

n_model = 1 (Beam), n_sg = 2 (2D cross-section). Material rotation is the per-element 3x3 frame from the YAML
(elem_rotation); there is no separate per-domain fibre angle (mat_angles = 0).
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from functools import partial
from scipy.sparse import csr_matrix
import pypardiso

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from basix import CellType, ElementFamily, LagrangeVariant, QuadratureType
if __package__ in (None, ""):          # allow running as a plain script too
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from opensg_jax.fe_jax.basis_quadrature import FiniteElementType, get_quadrature, eval_basis_and_derivatives
    from opensg_jax.fe_jax.setup import mesh_to_jax, mesh_to_periodic_sparse_assembly_map
    from opensg_jax.fe_jax.segment import read_solid_yaml
else:
    from .basis_quadrature import FiniteElementType, get_quadrature, eval_basis_and_derivatives
    from .setup import mesh_to_jax, mesh_to_periodic_sparse_assembly_map
    from .segment import read_solid_yaml


# ============================ material / constitutive (from Beam_solid.py) ============================
def build_orthotropic_C(params):
    E1, E2, E3, G12, G13, G23, v12, v13, v23 = params
    v21, v31, v32 = v12 * (E2 / E1), v13 * (E3 / E1), v23 * (E3 / E2)
    delta = (1.0 - v12 * v21 - v23 * v32 - v31 * v13 - 2.0 * v21 * v32 * v13)
    c11 = E1 * (1.0 - v23 * v32) / delta
    c22 = E2 * (1.0 - v13 * v31) / delta
    c33 = E3 * (1.0 - v12 * v21) / delta
    c12 = E1 * (v21 + v31 * v23) / delta
    c13 = E1 * (v31 + v21 * v32) / delta
    c23 = E2 * (v32 + v12 * v31) / delta
    return jnp.array([[c11, c12, c13, 0., 0., 0.],
                      [c12, c22, c23, 0., 0., 0.],
                      [c13, c23, c33, 0., 0., 0.],
                      [0., 0., 0., G23, 0., 0.],
                      [0., 0., 0., 0., G13, 0.],
                      [0., 0., 0., 0., 0., G12]])


def rotate_C_matrix(C, angle_deg):
    def do_rot(op):
        c_m, ang = op
        th = jnp.deg2rad(ang); c, s = jnp.cos(th), jnp.sin(th); cs = c * s
        R = jnp.array([[c**2, s**2, 0, 0, 0, -2 * cs], [s**2, c**2, 0, 0, 0, 2 * cs],
                       [0, 0, 1, 0, 0, 0], [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0],
                       [cs, -cs, 0, 0, 0, c**2 - s**2]])
        return R @ c_m @ R.T
    return jax.lax.cond(angle_deg == 0.0, lambda x: x[0], do_rot, (C, angle_deg))


def rotate_C_with_matrix(C, R_flat):
    R = R_flat.reshape(3, 3)
    r11, r12, r13 = R[0, 0], R[1, 0], R[2, 0]
    r21, r22, r23 = R[0, 1], R[1, 1], R[2, 1]
    r31, r32, r33 = R[0, 2], R[1, 2], R[2, 2]
    K = jnp.array([
        [r11**2, r12**2, r13**2, 2 * r12 * r13, 2 * r11 * r13, 2 * r11 * r12],
        [r21**2, r22**2, r23**2, 2 * r22 * r23, 2 * r21 * r23, 2 * r21 * r22],
        [r31**2, r32**2, r33**2, 2 * r32 * r33, 2 * r31 * r33, 2 * r31 * r32],
        [r21 * r31, r22 * r32, r23 * r33, r22 * r33 + r23 * r32, r21 * r33 + r23 * r31, r21 * r32 + r22 * r31],
        [r11 * r31, r12 * r32, r13 * r33, r12 * r33 + r13 * r32, r11 * r33 + r13 * r31, r11 * r32 + r12 * r31],
        [r11 * r21, r12 * r22, r13 * r23, r12 * r23 + r13 * r22, r11 * r23 + r13 * r21, r11 * r22 + r12 * r21]])
    return K @ C @ K.T


@partial(jax.jit, static_argnames=["num_quad_points"])
def get_heterogeneous_C_matrix(cell_domain_ids, num_quad_points, material_param, domain_angles, mat_seq, elem_rotation):
    # Identical to Beam_solid.py get_heterogeneous_C_matrix: build C per material -> apply the in-plane
    # ply angle (rotate_C_matrix) -> select per element -> rotate to the global frame by the per-element
    # frame (rotate_C_with_matrix). elem_rotation must already be a flat [e1,e2,e3] in the homogenizer's
    # (beam,x,y) axis order; segment.py performs the YAML (x,y,beam)->(beam,x,y) permutation.
    C_base_unique = jax.vmap(build_orthotropic_C)(material_param)
    C_domains = C_base_unique[mat_seq]
    C_domains_rotated = jax.vmap(rotate_C_matrix)(C_domains, domain_angles)
    C_final1 = C_domains_rotated[cell_domain_ids]
    C_final = jax.vmap(rotate_C_with_matrix)(C_final1, elem_rotation)
    return C_final, C_base_unique


# ============================ assembly (from Beam_solid.py, n_model=1) ============================
@partial(jax.jit, static_argnames=["N_primal", "n_model", "n_sg"])
def assemble_system_matrices(x_end, dphi_dxi_qnp, phi_qn, W_q, reduced_periodic_cells, N_primal, C_ess, n_model, n_sg):
    E_elem, N_nodes = x_end.shape[0], x_end.shape[1]
    mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
    mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
    mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)

    def get_element_matrices(x_nd, C_in):
        J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        det_J_q = jnp.abs(jnp.linalg.det(J_qdp)); dV_q = det_J_q * W_q
        inv_J = jnp.linalg.inv(J_qdp)
        dphi_dx = jnp.einsum("qpd,qnp->qnd", inv_J, dphi_dxi_qnp)
        x_q = jnp.einsum("qn,nd->qd", phi_qn, x_nd)
        Q = x_q.shape[0]
        C_q = jnp.repeat(C_in[None, :, :], Q, axis=0) if C_in.ndim == 2 else C_in
        if n_sg == 2:
            y2_q = x_q[:, 0]; y3_q = x_q[:, 1]
        else:
            y2_q = jnp.zeros_like(x_q[:, 0]); y3_q = x_q[:, 0]
        Ge = mask_ones_1d[None, ...] + mask_y2[None, ...] * y2_q[:, None, None] + mask_y3_1d[None, ...] * y3_q[:, None, None]

        def compute_eps_h(uh_flat):
            uh = uh_flat.reshape(N_nodes, 3); zeros = jnp.zeros_like(dphi_dx[..., 0]); D = dphi_dx.shape[-1]
            if D == 1:
                grad_phi = jnp.stack([zeros, zeros, dphi_dx[..., 0]], axis=-1)
            elif D == 2:
                grad_phi = jnp.stack([zeros, dphi_dx[..., 0], dphi_dx[..., 1]], axis=-1)
            else:
                grad_phi = dphi_dx
            grad_u = jnp.einsum("ni,qnj->qij", uh, grad_phi)
            return jnp.stack([grad_u[:, 0, 0], grad_u[:, 1, 1], grad_u[:, 2, 2],
                              grad_u[:, 1, 2] + grad_u[:, 2, 1], grad_u[:, 0, 2] + grad_u[:, 2, 0],
                              grad_u[:, 0, 1] + grad_u[:, 1, 0]], axis=1)

        def compute_eps_l(ul_flat):
            ul = ul_flat.reshape(N_nodes, 3); u_q = jnp.einsum("qn,ni->qi", phi_qn, ul); zeros = jnp.zeros_like(u_q[:, 0])
            return jnp.stack([u_q[:, 0], zeros, zeros, zeros, u_q[:, 2], u_q[:, 1]], axis=1)

        def compute_eps_e(ue_flat):
            return jnp.einsum("qip,p->qi", Ge, ue_flat)

        Ndofs = N_nodes * 3; Ndofs_e = Ge.shape[-1]
        zeros_u, zeros_e = jnp.zeros(Ndofs), jnp.zeros(Ndofs_e)

        def energy_hh(uh): e = compute_eps_h(uh); return 0.5 * jnp.einsum("qi,qij,qj,q->", e, C_q, e, dV_q)
        def energy_he(uh, ue): return jnp.einsum("qi,qij,qj,q->", compute_eps_h(uh), C_q, compute_eps_e(ue), dV_q)
        def energy_ee(ue): e = compute_eps_e(ue); return 0.5 * jnp.einsum("qi,qij,qj,q->", e, C_q, e, dV_q)
        D_hh_e = jax.hessian(energy_hh)(zeros_u)
        D_he_e = jax.jacfwd(jax.grad(energy_he, argnums=0), argnums=1)(zeros_u, zeros_e)
        D_ee_e = jax.hessian(energy_ee)(zeros_e)

        def energy_ll(ul): e = compute_eps_l(ul); return 0.5 * jnp.einsum("qi,qij,qj,q->", e, C_q, e, dV_q)
        def energy_hl(uh, ul): return jnp.einsum("qi,qij,qj,q->", compute_eps_h(uh), C_q, compute_eps_l(ul), dV_q)
        def energy_le(ul, ue): return jnp.einsum("qi,qij,qj,q->", compute_eps_l(ul), C_q, compute_eps_e(ue), dV_q)
        D_ll_e = jax.hessian(energy_ll)(zeros_u)
        D_hl_e = jax.jacfwd(jax.grad(energy_hl, argnums=0), argnums=1)(zeros_u, zeros_u)
        D_le_e = jax.jacfwd(jax.grad(energy_le, argnums=0), argnums=1)(zeros_u, zeros_e)
        return D_hh_e, D_he_e, D_ee_e, D_ll_e, D_hl_e, D_le_e, dphi_dx, Ge, dV_q

    vmap_out = jax.vmap(get_element_matrices, in_axes=(0, 0))(x_end, C_ess)
    dof_map = ((reduced_periodic_cells * 3).reshape(E_elem, N_nodes, 1) + jnp.array([0, 1, 2])).reshape(E_elem, N_nodes * 3)
    rows_sq = jnp.repeat(dof_map, N_nodes * 3, axis=1).ravel(); cols_sq = jnp.tile(dof_map, (1, N_nodes * 3)).ravel()
    D_hh_b, D_he_b, D_ee_b = vmap_out[0], vmap_out[1], vmap_out[2]
    Ndofs_e = D_he_b.shape[-1]
    rows_rect = jnp.repeat(dof_map, Ndofs_e, axis=1).ravel()
    cols_rect = jnp.tile(jnp.arange(Ndofs_e), (E_elem, N_nodes * 3)).ravel()
    Dhh = jsparse.COO((D_hh_b.ravel(), rows_sq, cols_sq), shape=(N_primal, N_primal))
    Dhe = jsparse.COO((D_he_b.ravel(), rows_rect, cols_rect), shape=(N_primal, Ndofs_e))
    Dee = jnp.sum(D_ee_b, axis=0)
    dehomo_data = {"dphi_dx": vmap_out[-3], "Ge": vmap_out[-2], "dV_q": vmap_out[-1]}
    omega = jnp.ptp(x_end[..., 0]) if n_sg == 3 else 1.0
    D_ll_b, D_hl_b, D_le_b = vmap_out[3], vmap_out[4], vmap_out[5]
    Dll = jsparse.COO((D_ll_b.ravel(), rows_sq, cols_sq), shape=(N_primal, N_primal))
    Dhl = jsparse.COO((D_hl_b.ravel(), rows_sq, cols_sq), shape=(N_primal, N_primal))
    Dle = jsparse.COO((D_le_b.ravel(), rows_rect, cols_rect), shape=(N_primal, Ndofs_e))
    return (Dhh._sort_indices(), Dhe._sort_indices(), Dee, Dll._sort_indices(), Dhl._sort_indices(),
            Dle._sort_indices(), omega, dehomo_data)


@partial(jax.jit, static_argnames=["N_primal_dofs"])
def build_lagrange_constraint_matrix(x_end, dphi_dxi_qnp, phi_qn, W_q, reduced_periodic_cells, N_primal_dofs):
    E, N_nodes = x_end.shape[0], x_end.shape[1]
    J_eqdp = jnp.einsum("end,qnp->eqdp", x_end, dphi_dxi_qnp)
    det_J_eq = jnp.linalg.det(J_eqdp); dV_eq = det_J_eq * W_q
    int_phi_en = jnp.einsum("qn,eq->en", phi_qn, dV_eq)
    x_eq = jnp.einsum("qn,end->eqd", phi_qn, x_end)
    y_eq, z_eq = x_eq[..., 0], x_eq[..., 1]
    int_y_phi_en = jnp.einsum("eq,qn,eq->en", y_eq, phi_qn, dV_eq)
    int_z_phi_en = jnp.einsum("eq,qn,eq->en", z_eq, phi_qn, dV_eq)
    zeros_en = jnp.zeros_like(int_phi_en)
    r0 = jnp.stack([int_phi_en, zeros_en, zeros_en], axis=-1).reshape(E, N_nodes * 3)
    r1 = jnp.stack([zeros_en, int_phi_en, zeros_en], axis=-1).reshape(E, N_nodes * 3)
    r2 = jnp.stack([zeros_en, zeros_en, int_phi_en], axis=-1).reshape(E, N_nodes * 3)
    r3 = jnp.stack([zeros_en, -int_z_phi_en, int_y_phi_en], axis=-1).reshape(E, N_nodes * 3)
    C_ett = jnp.stack([r0, r1, r2, r3], axis=1)
    dof_map_enu = jnp.stack([reduced_periodic_cells * 3 + 0, reduced_periodic_cells * 3 + 1,
                             reduced_periodic_cells * 3 + 2], axis=-1).reshape(E, N_nodes * 3)
    C_assembled = jnp.zeros((4, N_primal_dofs)).at[:, dof_map_enu].add(C_ett.transpose(1, 0, 2))
    return C_assembled


def compress_periodic_cells_jax(V, cells, dof_map, x_end, ndof_per_node=3):
    flat_x = x_end.reshape(-1, x_end.shape[-1]); flat_cells = cells.reshape(-1)
    global_points = jnp.zeros((V, x_end.shape[-1])).at[flat_cells].set(flat_x)
    master_nodes = dof_map[::ndof_per_node] // ndof_per_node
    unique_masters = jnp.unique(master_nodes); num_unique = int(unique_masters.shape[0])
    full_to_reduced = jnp.full(V, -1, dtype=jnp.int32).at[unique_masters].set(jnp.arange(num_unique, dtype=jnp.int32))
    reduced_periodic_cells = full_to_reduced[master_nodes[cells]]
    return reduced_periodic_cells, num_unique, global_points[unique_masters]


def solve_fluctuation_field(Dhh_sparse, RHS_dense, Dc_matrix, n_model):
    R_array = np.asarray(RHS_dense); actual_N = R_array.shape[0]; num_cases = R_array.shape[1]
    dhh_data = np.asarray(Dhh_sparse.data)
    dhh_row = np.asarray(Dhh_sparse.row); dhh_col = np.asarray(Dhh_sparse.col)
    Dc_dense = np.asarray(Dc_matrix)
    if Dc_dense.shape[0] == 4 and Dc_dense.shape[1] == actual_N:
        Dc_dense = Dc_dense.T
    num_constraints = Dc_dense.shape[1]
    Dc_scaled = Dc_dense * 1e8
    bl_row, bl_col = np.where(Dc_scaled.T); bl_data = Dc_scaled.T[bl_row, bl_col]; bl_row = bl_row + actual_N
    tr_row, tr_col = np.where(Dc_scaled); tr_data = Dc_scaled[tr_row, tr_col]; tr_col = tr_col + actual_N
    aug_row = np.concatenate([dhh_row, tr_row, bl_row]); aug_col = np.concatenate([dhh_col, tr_col, bl_col])
    aug_data = np.concatenate([dhh_data, tr_data, bl_data]); total_N = actual_N + num_constraints
    R_aug = np.vstack([R_array, np.zeros((num_constraints, num_cases), dtype=R_array.dtype)])
    row_counts = np.bincount(aug_row, minlength=total_N); empty_rows = np.where(row_counts == 0)[0]
    if len(empty_rows) > 0:
        aug_row = np.concatenate([aug_row, empty_rows]); aug_col = np.concatenate([aug_col, empty_rows])
        aug_data = np.concatenate([aug_data, np.ones_like(empty_rows, dtype=aug_data.dtype)]); R_aug[empty_rows, :] = 0.0
    A_augmented = csr_matrix((aug_data, (aug_row, aug_col)), shape=(total_N, total_N))
    V_aug = pypardiso.spsolve(A_augmented, R_aug)
    V0 = V_aug[:actual_N, :]; D1 = -(V0.T @ R_array)
    return jnp.array(V0), jnp.array(D1), A_augmented


@partial(jax.jit, static_argnames=["N_primal", "n_sg"])
def assemble_rigid_body_ops(x_unique, x_end, dphi_dxi_qnp, phi_qn, W_q, reduced_periodic_cells, N_primal, n_sg):
    dim = x_unique.shape[1]
    y_coords = jnp.zeros_like(x_unique[:, 0]) if dim == 1 else x_unique[:, dim - 2]
    z_coords = x_unique[:, dim - 1]; num_nodes = N_primal // 3
    psi_0 = jnp.tile(jnp.array([1., 0., 0.]), num_nodes); psi_1 = jnp.tile(jnp.array([0., 1., 0.]), num_nodes)
    psi_2 = jnp.tile(jnp.array([0., 0., 1.]), num_nodes)
    psi_3 = jnp.zeros(N_primal).at[1::3].set(-z_coords).at[2::3].set(y_coords)
    Psi = jnp.column_stack([psi_0, psi_1, psi_2, psi_3])
    E_elem, N_nodes = x_end.shape[0], x_end.shape[1]

    def element_Dc(x_nd):
        J = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp); det_J = jnp.abs(jnp.linalg.det(J)); dV = det_J * W_q
        inv_J = jnp.linalg.inv(J); dphi_dx = jnp.einsum("qpd,qnp->qnd", inv_J, dphi_dxi_qnp)
        idx_y = 0 if dphi_dx.shape[-1] == 2 else 1; idx_z = 1 if dphi_dx.shape[-1] == 2 else 2
        int_phi = jnp.einsum("qn,q->n", phi_qn, dV)
        int_dphi_dy = jnp.einsum("qn,q->n", dphi_dx[..., idx_y], dV)
        int_dphi_dz = jnp.einsum("qn,q->n", dphi_dx[..., idx_z], dV)
        c_e = jnp.zeros((4, N_nodes * 3))
        c_e = c_e.at[0, 0::3].set(int_phi).at[1, 1::3].set(int_phi).at[2, 2::3].set(int_phi)
        c_e = c_e.at[3, 1::3].set(int_dphi_dz).at[3, 2::3].set(-int_dphi_dy)
        return c_e
    c_batches = jax.vmap(element_Dc)(x_end)
    dof_map = ((reduced_periodic_cells * 3).reshape(E_elem, N_nodes, 1) + jnp.array([0, 1, 2])).reshape(E_elem, N_nodes * 3)
    Dc_T = jnp.zeros((4, N_primal)).at[:, dof_map.ravel()].add(c_batches.transpose(1, 0, 2).reshape(4, -1))
    return Psi, Dc_T.T


@jax.jit
def prepare_v1_rhs(V0, Dhl, Dll, Dle_dense, Psi, Dc):
    DhlV0 = Dhl @ V0; V0DllV0 = V0.T @ (Dll @ V0); DhlTV0Dle = Dhl.T @ V0 + Dle_dense
    b_unproj = DhlV0 - DhlTV0Dle
    inv_Dc_T_Psi = jnp.linalg.inv(Psi.T @ Dc); tmp = inv_Dc_T_Psi @ (Psi.T @ b_unproj)
    bb = (Dc @ tmp) - b_unproj
    return bb, DhlV0, DhlTV0Dle, V0DllV0


@jax.jit
def finalize_v1_and_compute_deff(V1s_raw, V0, D_eff, V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc):
    inv_DcT_Psi = jnp.linalg.inv(Dc.T @ Psi); V1s = V1s_raw - (Psi @ (inv_DcT_Psi @ (Dc.T @ V1s_raw)))
    Ainv = jnp.linalg.inv(D_eff)
    B_tim = DhlTV0Dle.T @ V0
    C_tim_unsym = V0DllV0 + V1s.T @ (DhlV0 + DhlTV0Dle); C_tim = 0.5 * (C_tim_unsym + C_tim_unsym.T)
    Q_base = jnp.array([[0., 0.], [0., 0.], [0., -1.], [1., 0.]], dtype=jnp.float64)
    Q_tim = Ainv @ Q_base
    Ginv = Q_tim.T @ (C_tim - B_tim.T @ Ainv @ B_tim) @ Q_tim; G_tim = jnp.linalg.inv(Ginv)
    Y_tim = B_tim.T @ Q_tim @ G_tim; A_tim = D_eff + Y_tim @ Ginv @ Y_tim.T
    D = jnp.zeros((6, 6), dtype=jnp.float64)
    D = D.at[0, 3:6].set(A_tim[0, 1:4]).at[0, 1:3].set(Y_tim[0, :]).at[0, 0].set(A_tim[0, 0])
    D = D.at[3:6, 3:6].set(A_tim[1:4, 1:4]).at[3:6, 1:3].set(Y_tim[1:4, :]).at[3:6, 0].set(A_tim[1:4, 0])
    D = D.at[1:3, 1:3].set(G_tim).at[1:3, 3:6].set(Y_tim.T[:, 1:4]).at[1:3, 0].set(Y_tim.T[:, 0])
    return D


# ============================ driver ============================
_CELL = {"triangle": CellType.triangle, "quad": CellType.quadrilateral}


def compute_timo_from_yaml(yaml_path, verbose=True):
    """Read a 2D-solid SG YAML and return the 6x6 Timoshenko stiffness (JAX MSG, VABS-equivalent)."""
    n_model = 1
    sg = read_solid_yaml(yaml_path)
    n_sg = sg["n_sg"]
    points = jnp.asarray(sg["points"]); cells = jnp.asarray(sg["cells"], dtype=jnp.uint64)
    V = points.shape[0]

    fe_type = FiniteElementType(cell_type=_CELL[sg["elem_type"]], family=ElementFamily.P, basis_degree=1,
                                lagrange_variant=LagrangeVariant.equispaced,
                                quadrature_type=QuadratureType.default, quadrature_degree=2)
    xi_qp, W_q = get_quadrature(fe_type=fe_type)
    phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe_type, xi_qp=xi_qp)
    Q = xi_qp.shape[0]
    x_end = mesh_to_jax(vertices=points, cells=cells)

    C_ess, _ = get_heterogeneous_C_matrix(jnp.asarray(sg["cell_domain_ids"]), Q,
                                          jnp.asarray(sg["material_param"]), jnp.asarray(sg["mat_angles"]),
                                          jnp.asarray(sg["mat_seq"]), jnp.asarray(sg["elem_rotation"]))

    cells_np = np.asarray(sg["cells"], dtype=np.uint64); points_np = np.asarray(sg["points"])
    _, dof_map_np = mesh_to_periodic_sparse_assembly_map(V, cells_np, points_np, n_model, atol=1e-6)
    reduced_cells, num_unique, x_unique = compress_periodic_cells_jax(V, cells_np, dof_map_np, x_end, ndof_per_node=3)
    N_primal = num_unique * 3
    if verbose:
        print("V=%d E=%d primal_dofs=%d Q=%d" % (V, x_end.shape[0], N_primal, Q), flush=True)

    Dhh, Dhe, Dee, Dll, Dhl, Dle, omega, _ = assemble_system_matrices(
        x_end, dphi_dxi_qnp, phi_qn, W_q, reduced_cells, N_primal, C_ess, n_model, n_sg)
    Psi, Dc = assemble_rigid_body_ops(x_unique, x_end, dphi_dxi_qnp, phi_qn, W_q, reduced_cells, N_primal, n_sg)

    RHS_V0 = -np.array(Dhe.todense())
    V0, D1_V0, A_aug = solve_fluctuation_field(Dhh, RHS_V0, Dc, n_model)
    Ceff = (Dee + D1_V0) / omega

    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)
    R_aug = np.concatenate([np.array(bb), np.zeros((4, np.array(bb).shape[1]))], axis=0)
    V1s_raw = jnp.array(pypardiso.spsolve(A_aug, R_aug)[:N_primal, :])
    C6 = np.asarray(finalize_v1_and_compute_deff(V1s_raw, V0, Ceff, V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc))
    return C6


if __name__ == "__main__":
    import sys
    C6 = compute_timo_from_yaml(sys.argv[1])
    lbl = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
    print("\nTimoshenko 6x6:")
    for i in range(6):
        print("  " + " ".join("% .4e" % C6[i, j] for j in range(6)))
    print("\ndiagonal:", " ".join("%s=%.4e" % (lbl[i], C6[i, i]) for i in range(6)))
