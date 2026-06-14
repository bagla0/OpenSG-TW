"""
MSG Shell Timoshenko Beam Homogenization — Quadratic Lagrange Elements

Kirchhoff shell model (eqs 3.30-3.32 of thesis) discretized along the
cross-section arc with 3-node quadratic Lagrange line elements (C0).

DOF per node: [w1, w2, w3]  (3 DOFs)
Element    : 3 nodes → 9 DOFs per element

Strain operators:
  Gamma_h — fluctuation strains (d/ds, d²/ds²)
  Gamma_e — macroscale beam strains (prismatic beam)
  Gamma_l — Timoshenko shear strains

All matrices are assembled via JAX energy-based autodiff (hessian / jacfwd).
The KKT system is solved with pypardiso (MKL PARDISO wrapper).
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
from scipy.sparse import csr_matrix
import pypardiso

jax.config.update('jax_default_matmul_precision', 'highest')
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)


# =============================================================================
# Quadrature and Shape Functions
# =============================================================================

def gauss_legendre_01(n_pts):
    """Gauss-Legendre quadrature points and weights on [0, 1]."""
    xi_ref, w_ref = np.polynomial.legendre.leggauss(n_pts)
    return jnp.array(0.5 * (xi_ref + 1.0)), jnp.array(0.5 * w_ref)


def quad_shape_functions(xi_q, L_elem):
    """Quadratic Lagrange shape functions on xi in [0, 1].

    Three-node element:
      N1 = 2*xi^2 - 3*xi + 1  (xi=0)
      N2 = -4*xi^2 + 4*xi      (xi=0.5)
      N3 = 2*xi^2 - xi          (xi=1)

    Returns
    -------
    phi_val : (Q, 3) values
    phi_d1  : (Q, 3) first physical derivatives  d/ds  = (1/L) d/dxi
    phi_d2  : (Q, 3) second physical derivatives d²/ds² = (1/L²) d²/dxi²
    """
    xi = xi_q
    xi2 = xi * xi
    L = L_elem

    phi_val = jnp.stack([
        2.0*xi2 - 3.0*xi + 1.0,
        -4.0*xi2 + 4.0*xi,
        2.0*xi2 - xi,
    ], axis=1)

    phi_d1 = jnp.stack([
        (4.0*xi - 3.0) / L,
        (-8.0*xi + 4.0) / L,
        (4.0*xi - 1.0) / L,
    ], axis=1)

    phi_d2 = jnp.stack([
        jnp.full_like(xi,  4.0) / L**2,
        jnp.full_like(xi, -8.0) / L**2,
        jnp.full_like(xi,  4.0) / L**2,
    ], axis=1)

    return phi_val, phi_d1, phi_d2


def compute_element_geometry(nodes, cells):
    """Arc-length L and tangent (xdot2, xdot3) per element.

    Uses the two end-corner nodes (first and last columns), so it works for
    both 3-node quadratic cells and 2-node Hermite cells.
    """
    n1, n2 = cells[:, 0], cells[:, -1]
    dy2 = nodes[n2, 0] - nodes[n1, 0]
    dy3 = nodes[n2, 1] - nodes[n1, 1]
    L_e = np.sqrt(dy2**2 + dy3**2)
    return jnp.array(L_e), jnp.array(dy2 / L_e), jnp.array(dy3 / L_e)


# =============================================================================
# DOF Maps (3 DOFs per node: [w1, w2, w3])
# =============================================================================

def build_periodic_dof_map(n_nodes, cells, is_closed=True):
    """DOF map redirecting the last node to node 0 for closed sections."""
    dofs_per_node = 3
    dof_map = np.arange(n_nodes * dofs_per_node, dtype=np.int64)
    if is_closed:
        last = n_nodes - 1
        for k in range(dofs_per_node):
            dof_map[last * dofs_per_node + k] = k
        n_unique = n_nodes - 1
    else:
        n_unique = n_nodes
    return dof_map, n_unique


def compress_dof_map(dof_map, cells, dofs_per_node=3):
    """Renumber full DOFs to unique (reduced) DOFs."""
    master_nodes = dof_map[::dofs_per_node] // dofs_per_node
    unique_masters = np.unique(master_nodes)
    n_primal = len(unique_masters) * dofs_per_node

    node_f2r = np.full(len(master_nodes), -1, dtype=np.int64)
    for i, m in enumerate(unique_masters):
        node_f2r[m] = i

    reduced_cells = node_f2r[master_nodes[cells]]

    full_to_reduced = np.zeros(len(dof_map), dtype=np.int64)
    for i in range(len(dof_map)):
        mn = dof_map[i] // dofs_per_node
        lk = dof_map[i] % dofs_per_node
        full_to_reduced[i] = node_f2r[mn] * dofs_per_node + lk

    return reduced_cells, full_to_reduced, n_primal


# =============================================================================
# System Matrix Assembly (energy-based autodiff, vmap over elements)
# =============================================================================

def assemble_system_matrices(
    nodes, cells, reduced_cells, ABD_elems, k22_elems,
    L_elems, xdot2_elems, xdot3_elems,
    xi_q, W_q, n_primal
):
    """Assemble Dhh, Dhe, Dee, Dll, Dhl, Dle via JAX energy autodiff.

    Each returned matrix is a JAX sparse COO (or dense for Dee).
    """
    E_elem = cells.shape[0]
    Q_pts = xi_q.shape[0]
    DPN = 3    # DOFs per node
    NPN = 3    # nodes per element
    NDE = 9    # DOFs per element

    def get_element_matrices(elem_data):
        n1c, nmc, n2c, ABD, k22, L_e, xd2, xd3 = elem_data

        phi_val, phi_d1, phi_d2 = quad_shape_functions(xi_q, L_e)
        dV_q = L_e * W_q

        x2_q = (1.0 - xi_q) * n1c[0] + xi_q * n2c[0]
        x3_q = (1.0 - xi_q) * n1c[1] + xi_q * n2c[1]
        ABD_q = jnp.repeat(ABD[None, :, :], Q_pts, axis=0)
        Rn = x2_q * xd3 - x3_q * xd2

        # gamma_e (MSG-TW shell, grad(x)=I, only k22 != 0): macro strains
        #   [axial, twist, bend2, bend3] -> shell strains [e11,e22,g12,k11,k22,k12]
        # Bending (cols 2,3) maps to shell bending k11 (row 3), NOT shear g12 (row 2).
        Ge = jnp.zeros((Q_pts, 6, 4))
        Ge = Ge.at[:, 0, 0].set(1.0)            # e11 from axial
        Ge = Ge.at[:, 0, 2].set(x3_q)           # e11 from bend2
        Ge = Ge.at[:, 0, 3].set(-x2_q)          # e11 from bend3
        Ge = Ge.at[:, 2, 1].set(Rn)             # g12 from twist
        Ge = Ge.at[:, 3, 2].set(xd2)            # k11 from bend2
        Ge = Ge.at[:, 3, 3].set(xd3)            # k11 from bend3
        Ge = Ge.at[:, 5, 1].set(-2.0 - k22 / 2.0 * Rn)   # k12 from twist

        i1 = jnp.array([0, 3, 6])   # w1 DOF indices within element
        i2 = jnp.array([1, 4, 7])   # w2
        i3 = jnp.array([2, 5, 8])   # w3

        def eps_h(u):
            dw1 = phi_d1 @ u[i1];  dw2 = phi_d1 @ u[i2];  dw3 = phi_d1 @ u[i3]
            d2w2 = phi_d2 @ u[i2]; d2w3 = phi_d2 @ u[i3]
            return jnp.stack([
                jnp.zeros(Q_pts),
                xd2 * dw2 + xd3 * dw3,
                dw1,
                jnp.zeros(Q_pts),
                -k22*xd2*dw2 + xd3*d2w2 - k22*xd3*dw3 - xd2*d2w3,
                (k22 / 2.0) * dw1,
            ], axis=1)

        def eps_l(u):
            w1p = phi_val @ u[i1]; w2p = phi_val @ u[i2]; w3p = phi_val @ u[i3]
            dw2p = phi_d1 @ u[i2]; dw3p = phi_d1 @ u[i3]
            # gamma_l (MSG-TW shell, grad(x)=I => d_i = e_i, only k22 != 0):
            #   L1 = w.e1, L3 = w.e2, L6 = 2*(dw/ds).e3 - (k22/2)*(w.e2); L2=L4=L5=0.
            # The shear-warping term lives in the kappa12 slot (index 5), NOT kappa22.
            return jnp.stack([
                w1p,
                jnp.zeros(Q_pts),
                xd2 * w2p + xd3 * w3p,
                jnp.zeros(Q_pts),
                jnp.zeros(Q_pts),
                2.0*xd3*dw2p - 2.0*xd2*dw3p - (k22/2.0)*(xd2*w2p + xd3*w3p),
            ], axis=1)

        def eps_e(ue):
            return jnp.einsum("qip,p->qi", Ge, ue)

        z_u = jnp.zeros(NDE)
        z_e = jnp.zeros(4)

        q = lambda e, eps: 0.5 * jnp.einsum("qi,qij,qj,q->", eps(e), ABD_q, eps(e), dV_q)
        b = lambda ea, fa, eb, fb: jnp.einsum("qi,qij,qj,q->", fa(ea), ABD_q, fb(eb), dV_q)

        D_hh = jax.hessian(lambda u: q(u, eps_h))(z_u)
        D_he = jax.jacfwd(jax.grad(lambda u, ue: b(u, eps_h, ue, eps_e), 0), 1)(z_u, z_e)
        D_ee = jax.hessian(lambda ue: q(ue, eps_e))(z_e)
        D_ll = jax.hessian(lambda u: q(u, eps_l))(z_u)
        D_hl = jax.jacfwd(jax.grad(lambda u, ul: b(u, eps_h, ul, eps_l), 0), 1)(z_u, z_u)
        D_le = jax.jacfwd(jax.grad(lambda u, ue: b(u, eps_l, ue, eps_e), 0), 1)(z_u, z_e)

        return D_hh, D_he, D_ee, D_ll, D_hl, D_le, Ge, dV_q

    n1_all  = jnp.array(nodes[cells[:, 0]])
    nm_all  = jnp.array(nodes[cells[:, 1]])
    n2_all  = jnp.array(nodes[cells[:, 2]])

    out = jax.vmap(get_element_matrices)(
        (n1_all, nm_all, n2_all, ABD_elems, k22_elems,
         L_elems, xdot2_elems, xdot3_elems)
    )
    D_hh_b, D_he_b, D_ee_b, D_ll_b, D_hl_b, D_le_b = out[:6]

    rc = jnp.array(reduced_cells)
    dof_elem = (
        (rc * DPN).reshape(E_elem, NPN, 1) + jnp.arange(DPN)
    ).reshape(E_elem, NDE)

    rows_sq   = jnp.repeat(dof_elem, NDE, axis=1).ravel()
    cols_sq   = jnp.tile(dof_elem, (1, NDE)).ravel()
    rows_rect = jnp.repeat(dof_elem, 4, axis=1).ravel()
    cols_rect = jnp.tile(jnp.arange(4), (E_elem, NDE)).ravel()

    def coo_sq(data):
        return jsparse.COO((data.ravel(), rows_sq, cols_sq),
                           shape=(n_primal, n_primal))

    def coo_rect(data):
        return jsparse.COO((data.ravel(), rows_rect, cols_rect),
                           shape=(n_primal, 4))

    return (
        coo_sq(D_hh_b)._sort_indices(),
        coo_rect(D_he_b)._sort_indices(),
        jnp.sum(D_ee_b, axis=0),
        coo_sq(D_ll_b)._sort_indices(),
        coo_sq(D_hl_b)._sort_indices(),
        coo_rect(D_le_b)._sort_indices(),
    )


# =============================================================================
# Lagrange Constraints  <w1>=<w2>=<w3>=<y3*w2-y2*w3>=0
# =============================================================================

def build_lagrange_constraints(
    nodes, cells, reduced_cells, L_elems, xi_q, W_q, n_primal
):
    """Build 4 x N_primal integral constraint matrix.

    Constraints (matching the OpenSG solid KKT, see Beam_solid.py):
      <w1> = <w2> = <w3> = 0                        (3 translations)
      INT( dw2/ds * xd3 - dw3/ds * xd2 ) ds = 0     (twist, DERIVATIVE form)

    The twist row is the 1D-shell image (Green's theorem) of the solid
    constraint INT(dw2/dz - dw3/dy) dA.  Using the derivative form (not the
    value form INT(y3*w2 - y2*w3)) makes the V1 right-hand side b orthogonal
    to the rigid-body kernel Psi, so the Timoshenko solve needs NO interior
    penalty -- the (large but harmless) C0 warping null components do not
    pollute C_tim.  ``dw/ds`` here is the physical arc derivative.
    """
    E_elem = cells.shape[0]
    DPN = 3
    NDE = 9

    def elem_c(n1c, n2c, L_e, _):
        phi_val, phi_d1, _ = quad_shape_functions(xi_q, L_e)
        dV_q = L_e * W_q
        xd2_e = (n2c[0] - n1c[0]) / L_e
        xd3_e = (n2c[1] - n1c[1]) / L_e

        ip  = jnp.einsum("qn,q->n", phi_val, dV_q)   # INT phi
        idp = jnp.einsum("qn,q->n", phi_d1, dV_q)    # INT dphi/ds

        c = jnp.zeros((4, NDE))
        c = c.at[0, jnp.array([0, 3, 6])].set(ip)
        c = c.at[1, jnp.array([1, 4, 7])].set(ip)
        c = c.at[2, jnp.array([2, 5, 8])].set(ip)
        c = c.at[3, jnp.array([1, 4, 7])].set(xd3_e * idp)   # dw2/ds * xd3
        c = c.at[3, jnp.array([2, 5, 8])].set(-xd2_e * idp)  # -dw3/ds * xd2
        return c

    rc = jnp.array(reduced_cells)
    batches = jax.vmap(elem_c)(
        jnp.array(nodes[cells[:, 0]]),
        jnp.array(nodes[cells[:, 2]]),
        L_elems, rc
    )

    dof_elem = (
        (rc * DPN).reshape(E_elem, 3, 1) + jnp.arange(DPN)
    ).reshape(E_elem, NDE)

    C_global = jnp.zeros((4, n_primal))
    C_global = C_global.at[:, dof_elem].add(batches.transpose(1, 0, 2))
    return C_global


# =============================================================================
# C0 Interior Penalty -- OPTIONAL fallback (not used by default)
# =============================================================================

def build_interior_penalty(cells, reduced_cells, L_elems, n_primal,
                           is_closed=True, penalty=None, abd_elems=None):
    """OPTIONAL C0 interior penalty on jump(dw/ds) across element nodes (CSR).

    Not needed in the default pipeline: the derivative-form twist constraint
    (``build_lagrange_constraints``) already keeps the V1 right-hand side
    orthogonal to the rigid kernel, so the C0 warping null modes stay
    harmless.  Kept as a fallback / cross-check that replicates OpenSG's
    ``deri_constraint`` C0-IP (Nitsche) term.  Per interior facet/component::

        Kp = -outer(c, a) - outer(a, c) + alpha * outer(a, a)

    with ``a = jump(dw/ds)``, ``c = avg(d2w/ds2)``,
    ``alpha = 10 * penalty / h_avg**2``.

    The ``penalty`` magnitude is made general across cross-sections/materials:
    if not given it is derived from the input ABD (the largest membrane
    stiffness ``A11`` over all layups), so it always tracks the stiffness
    scale instead of a hard-coded 1e9.
    """
    if penalty is None:
        if abd_elems is not None:
            penalty = float(np.max(np.abs(np.asarray(abd_elems)[:, 0, 0])))
        else:
            penalty = 1e9
    L = np.asarray(L_elems)
    rc = np.asarray(reduced_cells)        # rc[e] = [corner0, mid, corner1]
    n_elem = cells.shape[0]
    n_facets = n_elem if is_closed else n_elem - 1

    def d_ends(Le, tau):                  # quad-Lagrange N',N'' at element end
        d1 = np.array([4*tau - 3, -8*tau + 4, 4*tau - 1]) / Le
        d2 = np.array([4.0, -8.0, 4.0]) / Le**2
        return d1, d2

    rows, cols, data = [], [], []
    for n in range(n_facets):
        eL, eR = n, (n + 1) % n_elem
        h_avg = 0.5 * (L[eL] + L[eR])
        alpha = 10.0 * penalty / h_avg**2
        d1L, d2L = d_ends(L[eL], 1.0)     # left element, right end (xi=1)
        d1R, d2R = d_ends(L[eR], 0.0)     # right element, left end (xi=0)
        a = np.concatenate([d1L, -d1R])           # jump(dw/ds)
        c = np.concatenate([0.5 * d2L, 0.5 * d2R])  # avg(d2w/ds2)
        Kp = -np.outer(c, a) - np.outer(a, c) + alpha * np.outer(a, a)
        for k in range(3):                # each displacement component
            gdof = np.concatenate([rc[eL] * 3 + k, rc[eR] * 3 + k])
            for ii in range(6):
                rows.extend(gdof); cols.extend([gdof[ii]] * 6)
                data.extend(Kp[ii, :])
    return csr_matrix((data, (cols, rows)), shape=(n_primal, n_primal))


# =============================================================================
# Rigid Body Mode Matrix (Psi)
# =============================================================================

def build_psi_matrix(nodes, n_unique_nodes, n_primal):
    """N_primal x 4 matrix of rigid-body modes: 3 translations + twist."""
    DPN = 3
    psi = jnp.zeros((n_primal, 4))
    psi = psi.at[0::DPN, 0].set(1.0)   # w1 translation
    psi = psi.at[1::DPN, 1].set(1.0)   # w2 translation
    psi = psi.at[2::DPN, 2].set(1.0)   # w3 translation
    for i in range(n_unique_nodes):
        y2_i, y3_i = nodes[i, 0], nodes[i, 1]
        psi = psi.at[i*DPN + 1, 3].set(-y3_i)  # w2 = -y3 (twist)
        psi = psi.at[i*DPN + 2, 3].set( y2_i)  # w3 =  y2 (twist)
    return psi


# =============================================================================
# KKT Solver
# =============================================================================

def solve_fluctuation_field(Dhh_sparse, RHS_dense, Dc_matrix):
    """Solve KKT system [Dhh  Dc; Dc^T  0][V0; lam] = [RHS; 0].

    Uses pypardiso (Intel MKL PARDISO) for direct sparse solve.

    Returns
    -------
    V0          : (N_primal, n_cases) fluctuation field
    D1          : (n_cases, n_cases) = -(V0^T @ RHS)  [= V0^T @ F_load since RHS=-F]
    A_augmented : scipy CSR of the full KKT system (reused for Timoshenko RHS)
    """
    R = np.asarray(RHS_dense)
    N, n_cases = R.shape

    if not hasattr(Dhh_sparse, 'row'):
        Dhh_sparse = Dhh_sparse.tocoo()
    dhh_data = np.asarray(Dhh_sparse.data)
    dhh_row  = np.asarray(Dhh_sparse.row)
    dhh_col  = np.asarray(Dhh_sparse.col)

    Dc = np.asarray(Dc_matrix)
    if Dc.shape[0] == 4 and Dc.shape[1] == N:
        Dc = Dc.T
    n_con = Dc.shape[1]
    Dc_sc = Dc * 1e8

    bl_r, bl_c = np.where(Dc_sc.T);   bl_r = bl_r + N
    tr_r, tr_c = np.where(Dc_sc);     tr_c = tr_c + N

    aug_row  = np.concatenate([dhh_row, tr_r, bl_r])
    aug_col  = np.concatenate([dhh_col, tr_c, bl_c])
    aug_data = np.concatenate([dhh_data, Dc_sc.T[bl_r-N, bl_c-N],
                                          Dc_sc[tr_r, tr_c-N]])

    total = N + n_con
    R_aug = np.vstack([R, np.zeros((n_con, n_cases))])

    cnt = np.bincount(aug_row, minlength=total)
    emp = np.where(cnt == 0)[0]
    if len(emp):
        aug_row  = np.concatenate([aug_row, emp])
        aug_col  = np.concatenate([aug_col, emp])
        aug_data = np.concatenate([aug_data, np.ones(len(emp), dtype=aug_data.dtype)])
        R_aug[emp] = 0.0

    A_aug = csr_matrix((aug_data, (aug_row, aug_col)), shape=(total, total))
    V_aug = pypardiso.spsolve(A_aug, R_aug)

    V0 = V_aug[:N, :]
    D1 = -(V0.T @ R)
    return jnp.array(V0), jnp.array(D1), A_aug


# =============================================================================
# Timoshenko Enhancement
# =============================================================================

@jax.jit
def prepare_v1_rhs(V0, Dhl, Dll, Dle_dense, Psi, Dc):
    """Compute RHS for the V1 (shear warping) solve."""
    DhlV0 = Dhl @ V0
    V0DllV0 = V0.T @ (Dll @ V0)
    DhlTV0Dle = Dhl.T @ V0 + Dle_dense
    b_unproj = DhlV0 - DhlTV0Dle

    tmp = jnp.linalg.inv(Psi.T @ Dc) @ (Psi.T @ b_unproj)
    bb = (Dc @ tmp) - b_unproj
    return bb, DhlV0, DhlTV0Dle, V0DllV0


@jax.jit
def finalize_v1_and_compute_deff(V1s_raw, V0, D_eff, V0DllV0,
                                  DhlV0, DhlTV0Dle, Psi, Dc):
    """Project V1s and assemble 6x6 Timoshenko stiffness.

    Output ordering: [gamma11, gamma12, gamma13, kappa1, kappa2, kappa3]
    """
    tmp = jnp.linalg.inv(Dc.T @ Psi) @ (Dc.T @ V1s_raw)
    V1s = V1s_raw - (Psi @ tmp)

    Ainv = jnp.linalg.inv(D_eff)
    B_tim = DhlTV0Dle.T @ V0

    C_unsym = V0DllV0 + V1s.T @ (DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_unsym + C_unsym.T)

    Q_base = jnp.array([[0., 0.], [0., 0.], [0., -1.], [1., 0.]])
    Q_tim = Ainv @ Q_base
    Ginv = Q_tim.T @ (C_tim - B_tim.T @ Ainv @ B_tim) @ Q_tim
    G_tim = jnp.linalg.inv(Ginv)
    Y_tim = B_tim.T @ Q_tim @ G_tim
    A_tim = D_eff + Y_tim @ Ginv @ Y_tim.T

    # Pack into 6x6: rows/cols = [EA, GA12, GA13, GJ, EI2, EI3]
    S = jnp.zeros((6, 6))
    S = S.at[0, 0].set(A_tim[0, 0])
    S = S.at[0, 1:3].set(Y_tim[0, :])
    S = S.at[0, 3:6].set(A_tim[0, 1:4])
    S = S.at[1:3, 0].set(Y_tim.T[:, 0])
    S = S.at[1:3, 1:3].set(G_tim)
    S = S.at[1:3, 3:6].set(Y_tim.T[:, 1:4])
    S = S.at[3:6, 0].set(A_tim[1:4, 0])
    S = S.at[3:6, 1:3].set(Y_tim[1:4, :])
    S = S.at[3:6, 3:6].set(A_tim[1:4, 1:4])
    return S, B_tim, C_tim, V1s
