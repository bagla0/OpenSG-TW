"""msg_rm_plate.py -- MSG-based Reissner-Mindlin plate law + recovery (Yu-2002 construction).

Core OpenSG module, JAX implementation.  The through-thickness SG pipeline is one traced
function per (n_ply, n_per_layer, elem_order) BUCKET -- vmapped over elements inside (the
beam-solid pattern of ``solid_timo``: batched elemental D-matrices + scatter assembly) and
vmapped over LAYUPS by ``rm_plate_msg_batch`` (the analog of the solid's hex/tet
element-type buckets).  jit-compiled once per bucket (~1.7 s), then a few ms per laminate
in the batched path (measured 12x the per-laminate NumPy loop on the 10-wall st15 station,
83x at blade scale ~300 laminates).

Construction (equation numbers = Yu, Hodges & Volovoi, Computers & Structures 81:439-454,
2003; the same construction is IJSS 39:5185 Eqs. 49-55 and CMAME 191:5087 Eqs. 70-77):

  constraints    : both warping solves are the constrained minimization Eq. (34) solved
                   DIRECTLY as a Lagrange-multiplier (KKT) saddle-point system, so the
                   multiplier of Eq. (35) and the gauge <w_i> = 0 of Eqs. (31)/(39) are
                   exact by construction -- no penalty, no post-hoc projection.  The
                   operator is the same for every order, so it is factorized ONCE and the
                   18 unit load cases (6 plate strains + 2x6 strain gradients) reuse it
  zeroth order   : V0 warping, Eq. (39)  -> A6 (classical ABD, Eq. (40); must reproduce
                   compute_ABD_matrix exactly).  The multiplier vanishes identically here
                   (kernel^T D_he = 0; asserted in the test suite, not recomputed per call)
  first order    : gradient-driven warping columns V11, V12 (paper Eq. 45),
                   Eq. (45), driven by Dbar_a of Eqs. (43)-(44); here the multiplier is
                   genuinely nonzero and its content is the S_a of the RM projection
  second order   : gradient energy H (12x12 blocks [[B,C],[C^T,D]] of Eq. (47) over
                   [E,1; E,2], entering the energy Eq. (46))
  RM projection  : least-squares minimization of the residual U* (Eqs. (56)-(57)) over the
                   shear compliance X = G^{-1} (3) plus the relaxed-constraint constants
                   c_a (24, in-plane warping shifts only -- Yu Eq. (58) count; the w3-shift
                   columns are identically inert for monoclinic laminates)
                   -> the MSG transverse-shear stiffness  G_msg (2x2), Eq. (61)
  second order   : V2 warping columns V21/V22/V23 (Eq. 64), driven by the second plate-
                   strain gradients; energy content O((h/l)^4) so A6/G are untouched --
                   V2 exists purely so the recovery carries sigma33 at its leading order
  recovery       : through-thickness 3-D strain -- Eq. (63) with gradients E,1/E,2, or the
                   SECOND-order Eq. (66) when the E,11/E,12/E,22 gradients are also given
                   (msgrm_strain_at_depth); Eq. (65) warping displacement S(V0+V1bar+V2)
                   via msgrm_warping_at_depth.

Voigt strain order [11,22,33,23,13,12]; plate strain order E = [e11,e22,g12,k11,k22,k12];
transverse shear gamma = [2g13, 2g23].  x = through-thickness coordinate measured from the
reference plane set by ``fraction`` (0 = bottom/OML face = default, 0.5 = center, 1 = IML).

DIMENSION SUFFIXES on the internal arrays (the returned dict keeps the paper symbols):
    n  ndofs, the through-thickness warping dofs (3 per node)
    l  3*(p+1), the dofs of ONE element (elemental blocks before scatter)
    k  3, the kernel/psi dimension (the constant warping modes)
    v  6, 3-D Voigt strain components   [11,22,33,23,13,12]
    s  6, PLATE strain components       [e11,e22,g12,k11,k22,k12]  (also the number of
       unit load cases per gradient direction: eps,a has 6 components)
    i  2, the IN-PLANE warping components (w1,w2) that carry Yu's relaxed constants
    g  2, the transverse shears          [2g13, 2g23]
    t  12, the stacked gradient vector [eps,1 ; eps,2] -> the residual operator is (t,t)
    q  144, the raveled entries of that (t,t) operator
    m  27, the least-squares unknowns [X(3), c1(12), c2(12)]
e.g. D_he_ns is <Gamma_h^T C Gamma_eps> (n x s), V11_ns a first-order warping column
block, c1_is the (2 x 6) relaxed constants, L1_ns = kernel_nk @ c1_ks its ndofs x 6 field.

Validation lives in ``tests/test_msg_rm_plate.py`` (pytest, or run it directly)
"""
import numpy as np
import jax
import jax.numpy as jnp

from jax.scipy.linalg import lu_factor, lu_solve

from .msg_materials import rotated_stiffness_6x6, _plate_B


def _lagrange_N(nodes_xi, xi):
    npn = len(nodes_xi)
    N = np.ones(npn)
    for i in range(npn):
        for j in range(npn):
            if j != i:
                N[i] *= (xi - nodes_xi[j]) / (nodes_xi[i] - nodes_xi[j])
    return N


def _grad_ops(nodes_xi, xi):
    """Gamma_l1, Gamma_l2 (6 x 3*(p+1)): strain contribution of the IN-PLANE gradient of the warping.
    w,1: e11 += w1,1 ; 2g13 += w3,1 ; g12 += w2,1
    w,2: e22 += w2,2 ; 2g23 += w3,2 ; g12 += w1,2
    """
    N = _lagrange_N(nodes_xi, xi)
    npn = len(nodes_xi)
    Gamma_l1 = np.zeros((6, 3 * npn)); Gamma_l2 = np.zeros((6, 3 * npn))
    for n in range(npn):
        Gamma_l1[0, 3 * n + 0] = N[n]      # eps11 <- w1,1
        Gamma_l1[4, 3 * n + 2] = N[n]      # 2g13  <- w3,1
        Gamma_l1[5, 3 * n + 1] = N[n]      # g12   <- w2,1
        Gamma_l2[1, 3 * n + 1] = N[n]      # eps22 <- w2,2
        Gamma_l2[3, 3 * n + 2] = N[n]      # 2g23  <- w3,2
        Gamma_l2[5, 3 * n + 0] = N[n]      # g12   <- w1,2
    return Gamma_l1, Gamma_l2


# Gamma_eps = E0 + x3*E1 (Eq. 11 direct plate-strain term, code strain order)
_E0 = np.zeros((6, 6)); _E0[0, 0] = _E0[1, 1] = _E0[5, 2] = 1.0
_E1 = np.zeros((6, 6)); _E1[0, 3] = _E1[1, 4] = _E1[5, 5] = 1.0

_BUCKETS = {}

# Relative singular-value cutoff for the truncated-SVD minimum-norm solve of the U* least
# squares (the "sigma_i = 0" test of Ascher & Greif Sec. 8.2).  The U* system is rank 26 of
# 27; its null singular value sits ~1e-16 of the largest, so anything in 1e-14..1e-10 works.
_LS_RCOND = 1e-12


def _bucket(nlay, n_per_layer, p):
    """Static structures + jitted kernels for one (n_ply, n_per_layer, elem_order) bucket."""
    key = (int(nlay), int(n_per_layer), int(p))
    if key in _BUCKETS:
        return _BUCKETS[key]
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    xi_g, w_g = np.polynomial.legendre.leggauss(max(3, p + 1))
    # static reference operator tables per Gauss point (element-independent):
    #   [Gamma_h S] scales as 2/he -> store at he=2 (unit scale); [Gamma_l_a S] has no he
    Gh_ref = np.stack([_plate_B(nodes_xi, xi, 2.0) for xi in xi_g])
    Gl1_ref, Gl2_ref = zip(*[_grad_ops(nodes_xi, xi) for xi in xi_g])
    Gl1_ref = np.stack(Gl1_ref); Gl2_ref = np.stack(Gl2_ref)

    n_elem = nlay * n_per_layer
    n_node = p * n_elem + 1
    ndofs = 3 * n_node
    dofs_e = np.stack([np.arange(3 * p * e, 3 * p * e + 3 * (p + 1)) for e in range(n_elem)])
    elem_layer = np.repeat(np.arange(nlay), n_per_layer)
    sub_of_elem = np.arange(n_elem) % n_per_layer

    # kernel of D_hh = the 3 constant (rigid-translation) through-thickness modes; the
    # paper's psi (Eq. 31, psi^T H psi = I with H = <S^T S>) is kernel/sqrt(h), built
    # in-trace because h depends on the laminate.
    kernel_nk = np.zeros((ndofs, 3))
    kernel_nk[0::3, 0] = 1.0; kernel_nk[1::3, 1] = 1.0; kernel_nk[2::3, 2] = 1.0
    nodes_of_e = np.stack([np.arange(p * e, p * e + p + 1) for e in range(n_elem)])
    S_int_ref = np.array([sum(w_g[q] * _lagrange_N(nodes_xi, xi_g[q])[a] for q in range(len(xi_g)))
                          for a in range(p + 1)])   # int N_a dxi over [-1,1] (for H = <S^T S>)

    # D_1, D_2 (Eq. 51): Boolean selectors of eps = R - D_a gamma,_a, code strain order
    D1_sg = np.zeros((6, 2)); D2_sg = np.zeros((6, 2))
    D1_sg[3, 0] = 1.0; D1_sg[5, 1] = 1.0     # k11 <- 2g13,1 ; k12 <- 2g23,1
    D2_sg[4, 1] = 1.0; D2_sg[5, 0] = 1.0     # k22 <- 2g23,2 ; k12 <- 2g13,2

    # unit directions of the 27 LS unknowns [x11,x12,x22, c1(12), c2(12)] (Eq. 58: 3 + 24)
    units = []
    for j in range(27):
        pj = np.zeros(27); pj[j] = 1.0
        units.append((np.array([[pj[0], pj[1]], [pj[1], pj[2]]]),
                      pj[3:15].reshape(2, 6), pj[15:27].reshape(2, 6)))

    jGh = jnp.asarray(Gh_ref); jGl1 = jnp.asarray(Gl1_ref); jGl2 = jnp.asarray(Gl2_ref)
    jE0 = jnp.asarray(_E0); jE1 = jnp.asarray(_E1)
    jdofs = jnp.asarray(dofs_e); jlayer = jnp.asarray(elem_layer); jsub = jnp.asarray(sub_of_elem)
    jkernel_nk = jnp.asarray(kernel_nk)
    jnodes_e = jnp.asarray(nodes_of_e); jS_int = jnp.asarray(S_int_ref)
    jD1_sg = jnp.asarray(D1_sg); jD2_sg = jnp.asarray(D2_sg)
    junits = [tuple(jnp.asarray(u) for u in unit) for unit in units]
    jw_g = jnp.asarray(w_g); jxi_g = jnp.asarray(xi_g)

    def single(thick, C_layers, fraction):
        """One laminate: thick (nlay,), C_layers (nlay,6,6) rotated 3-D stiffness, fraction scalar."""
        h = jnp.sum(thick)
        z_ref = fraction * h                                   # reference plane (0=OML .. 1=IML)
        he = jnp.repeat(thick / n_per_layer, n_per_layer)      # (n_elem,) element thickness
        layer_bot = jnp.concatenate([jnp.zeros(1), jnp.cumsum(thick)])
        x_left = layer_bot[jlayer] + jsub * he - z_ref         # (n_elem,) element bottom x3
        Ck = C_layers[jlayer]                                  # (n_elem,6,6)

        def elem_D_mats(he_e, xl_e, C_e):
            """Elemental Eq.-30 integrals (local 12x12 / 12x6 / 6x6 blocks; l = local dofs)."""
            D_hh_ll = jnp.zeros((3 * (p + 1), 3 * (p + 1)))
            D_he_ls = jnp.zeros((3 * (p + 1), 6)); D_ee_ss = jnp.zeros((6, 6))
            D_hl1_ll = jnp.zeros_like(D_hh_ll); D_hl2_ll = jnp.zeros_like(D_hh_ll)
            D_l1l1_ll = jnp.zeros_like(D_hh_ll); D_l1l2_ll = jnp.zeros_like(D_hh_ll)
            D_l2l2_ll = jnp.zeros_like(D_hh_ll)
            D_l1e_ls = jnp.zeros_like(D_he_ls); D_l2e_ls = jnp.zeros_like(D_he_ls)
            for q in range(len(xi_g)):                          # unrolled Gauss loop
                dw = 0.5 * he_e * jw_g[q]
                Gamma_h_vl = (2.0 / he_e) * jGh[q]              # [Gamma_h S]: d/dx3 strains
                Gamma_l1_vl = jGl1[q]; Gamma_l2_vl = jGl2[q]    # [Gamma_l1 S], [Gamma_l2 S]
                x_q = xl_e + he_e * 0.5 * (1.0 + jxi_g[q])
                Gamma_eps_vs = jE0 + x_q * jE1                  # Gamma_eps: e + x3*k rows
                D_hh_ll += Gamma_h_vl.T @ C_e @ Gamma_h_vl * dw
                D_he_ls += Gamma_h_vl.T @ C_e @ Gamma_eps_vs * dw
                D_ee_ss += Gamma_eps_vs.T @ C_e @ Gamma_eps_vs * dw
                D_hl1_ll += Gamma_h_vl.T @ C_e @ Gamma_l1_vl * dw
                D_hl2_ll += Gamma_h_vl.T @ C_e @ Gamma_l2_vl * dw
                D_l1l1_ll += Gamma_l1_vl.T @ C_e @ Gamma_l1_vl * dw
                D_l1l2_ll += Gamma_l1_vl.T @ C_e @ Gamma_l2_vl * dw
                D_l2l2_ll += Gamma_l2_vl.T @ C_e @ Gamma_l2_vl * dw
                D_l1e_ls += Gamma_l1_vl.T @ C_e @ Gamma_eps_vs * dw
                D_l2e_ls += Gamma_l2_vl.T @ C_e @ Gamma_eps_vs * dw
            return (D_hh_ll, D_he_ls, D_ee_ss, D_hl1_ll, D_hl2_ll,
                    D_l1l1_ll, D_l1l2_ll, D_l2l2_ll, D_l1e_ls, D_l2e_ls)

        # vmapped elemental blocks (the beam-solid pattern), _b = batched over elements
        (D_hh_b, D_he_b, D_ee_b, D_hl1_b, D_hl2_b,
         D_l1l1_b, D_l1l2_b, D_l2l2_b, D_l1e_b, D_l2e_b) = jax.vmap(elem_D_mats)(he, x_left, Ck)

        rows = jdofs[:, :, None]; cols = jdofs[:, None, :]

        def scat_nn(Be):                                        # scatter-assemble (ndofs,ndofs)
            return jnp.zeros((ndofs, ndofs)).at[rows, cols].add(Be)

        def scat_ns(Be):                                        # scatter-assemble (ndofs,6)
            return jnp.zeros((ndofs, 6)).at[jdofs].add(Be)

        # assembled SG matrices, paper Eq. 30 names (dimension suffixes: see module docstring)
        D_hh_nn = scat_nn(D_hh_b)                               # E     = <Gamma_h^T C Gamma_h>
        D_he_ns = scat_ns(D_he_b)                               # D_he  = <Gamma_h^T C Gamma_eps>
        D_ee_ss = jnp.sum(D_ee_b, axis=0)                       # D_ee  = <Gamma_eps^T C Gamma_eps>
        D_hl1_nn = scat_nn(D_hl1_b); D_hl2_nn = scat_nn(D_hl2_b)
        D_l1l1_nn = scat_nn(D_l1l1_b); D_l1l2_nn = scat_nn(D_l1l2_b); D_l2l2_nn = scat_nn(D_l2l2_b)
        D_l1e_ns = scat_ns(D_l1e_b); D_l2e_ns = scat_ns(D_l2e_b)

        # ---- the constraint machinery of Eqs. (31)-(39) ----
        # H = <S^T S> enters only through H @ psi: the through-thickness integration
        # functional (per-node weights int N_a dx3, one copy per warping component).
        w_node = jnp.zeros(n_node).at[jnodes_e].add(0.5 * he[:, None] * jS_int)
        w_dof_n = jnp.repeat(w_node, 3)                        # <.> weights on the dofs
        psi_nk = jkernel_nk / jnp.sqrt(h)                      # Eq. (31): psi^T H psi = I3
        Hpsi_nk = w_dof_n[:, None] * psi_nk                    # H @ psi  (n, k)
        # Eq. (34) solved DIRECTLY as the Lagrange-multiplier saddle-point system:
        #   [[D_hh, H psi], [psi^T H, 0]] [V; Lam] = [-rhs; 0]
        # the multiplier row reproduces Eq. (35) automatically (Lam = -psi^T rhs: zero at
        # zeroth order since Gamma_h of a constant vanishes, = -S_a/sqrt(h) at first
        # order), and the constraint row enforces <w_i> = 0 exactly, so Eq. (39) is
        # built into the solve.  scl balances the two blocks for conditioning only.
        scl = jnp.max(jnp.abs(jnp.diag(D_hh_nn)))
        KKT = jnp.block([[D_hh_nn, scl * Hpsi_nk],
                         [scl * Hpsi_nk.T, jnp.zeros((3, 3))]])
        # ONE factorization shared by all 18 unit load cases (the zeroth- and first-order
        # problems differ only in the forcing).  All three formulations give bit-identical
        # results; timings under the outer laminate vmap (min of paired interleaved trials,
        # shared machine) put this one ~15-25% ahead of two independent multi-RHS solves,
        # with one lu_solve vmapped per load-case column the slowest of the three -- so the
        # case columns are kept as a matrix (level-3 BLAS) rather than vmapped individually.
        lu_piv = lu_factor(KKT)

        def solve_constrained(rhs):
            """All load cases at once: rhs is (ndofs, n_case), one column per unit plate
            strain (zeroth order) or strain-gradient component (first order)."""
            return lu_solve(lu_piv, jnp.vstack([-rhs, jnp.zeros((3, rhs.shape[1]))]))[:ndofs]

        # zeroth order (Eq. 39-40): V0 columns per unit plate strain; A6 = classical ABD.
        # The multiplier vanishes identically here: kernel^T D_he = <(Gamma_h kernel)^T C
        # Gamma_eps> = 0 because Gamma_h of a through-thickness constant is zero.  So no
        # correction to the forcing is needed at zeroth order (kept as a check below).
        # The CONSTRAINT row is still load-bearing: it selects the unique V0 with <w>=0,
        # and that gauge propagates into D_abar (D_hla @ kernel != 0) and hence into G.
        V0_ns = solve_constrained(D_he_ns)
        A6_ss = D_ee_ss + V0_ns.T @ D_he_ns


        D1bar_ns = (D_hl1_nn - D_hl1_nn.T) @ V0_ns - D_l1e_ns
        D2bar_ns = (D_hl2_nn - D_hl2_nn.T) @ V0_ns - D_l2e_ns

        V1_n2s = solve_constrained(jnp.concatenate([D1bar_ns, D2bar_ns], axis=1))  # 12 at once
        V11_ns = V1_n2s[:, :6]; V12_ns = V1_n2s[:, 6:]

        # gradient energy blocks B, C, D of Eq. (47) -> H = [[B,C],[C^T,D]] over [E,1; E,2]
        H11_ss = V0_ns.T @ D_l1l1_nn @ V0_ns + D1bar_ns.T @ V11_ns
        H12_ss = V0_ns.T @ D_l1l2_nn @ V0_ns + 0.5 * (D1bar_ns.T @ V12_ns + V11_ns.T @ D2bar_ns)
        H22_ss = V0_ns.T @ D_l2l2_nn @ V0_ns + D2bar_ns.T @ V12_ns

        H11_ss = 0.5 * (H11_ss + H11_ss.T); H22_ss = 0.5 * (H22_ss + H22_ss.T)
        H_tt = jnp.block([[H11_ss, H12_ss], [H12_ss.T, H22_ss]])

        S1_is = (jkernel_nk.T @ D1bar_ns)[:2]
        S2_is = (jkernel_nk.T @ D2bar_ns)[:2]
        AD1_sg = A6_ss @ jD1_sg; AD2_sg = A6_ss @ jD2_sg

        def blocks(X_gg, c1_is, c2_is):
            """[[Bhat,Chat],[Chat^T,Dhat]] of Eqs. (57)+(60); its 78 entries must -> 0.

            Per block: H.. = B/C/D of Eq. (47), AD_a X AD_b^T = A D_a G^-1 D_b^T A of
            Eq. (57), and the c-terms are the Eq. (60) relaxation via L_a = kernel c_a,
            S_a = kernel^T Dbar_a  ->  L_a^T Dbar_b = c_a^T S_b.

            DIAGONAL blocks carry Eq. (60)'s 2 L_1^T Dbar_1 as its SYMMETRIC representative
            c1^T S1 + S1^T c1 = 2 sym(c1^T S1).  Do not "correct" this to 2*c1.T@S1: the raw
            product is non-symmetric (order-1 asymmetry), and although the two agree inside
            the quadratic form eps,1^T(.)eps,1 that U* actually is, the Frobenius objective
            below DOES see the antisymmetric part.  Feeding the raw form in lets the 24
            constants chase a residual with no energetic meaning: measured G error up to 14%
            and Ustar_rel inflated up to 10x.  The OFF-diagonal Chat is deliberately NOT
            symmetrized -- it matches Eq. (60) literally, because it sits in the bilinear
            2 R,1^T Chat R,2 between two different vectors, where the asymmetry is real."""
            Bs_ss = H11_ss + AD1_sg @ X_gg @ AD1_sg.T + c1_is.T @ S1_is + S1_is.T @ c1_is
            Cs_ss = H12_ss + AD1_sg @ X_gg @ AD2_sg.T + c1_is.T @ S2_is + S1_is.T @ c2_is
            Ds_ss = H22_ss + AD2_sg @ X_gg @ AD2_sg.T + c2_is.T @ S2_is + S2_is.T @ c2_is
            return jnp.block([[Bs_ss, Cs_ss], [Cs_ss.T, Ds_ss]])

        # linear LS ("78 equations, 27 unknowns", text after Eq. 57): columns by unit probing
        M0_tt = blocks(jnp.zeros((2, 2)), jnp.zeros((2, 6)), jnp.zeros((2, 6)))  # == H exactly
        b0_q = -M0_tt.ravel()
        Amat_qm = jnp.stack([blocks(*unit).ravel() + b0_q for unit in junits], axis=1)
        # Column equilibration: the X-columns scale like A^2 (~1e13) and the c-columns like
        # S (~1e9), a ~1e4 spread that pushes cond(Amat) to ~5e19.  Scaling each column to
        # unit norm brings it to ~8e15.  Only the redundant constants move; X is invariant.
        cs_m = jnp.linalg.norm(Amat_qm, axis=0)
        cs_m = jnp.where(cs_m == 0, 1.0, cs_m)

        U_qm, sig_m, Vt_mm = jnp.linalg.svd(Amat_qm / cs_m, full_matrices=False)
        sig_ok = sig_m > _LS_RCOND * sig_m[0]
        # guard the reciprocal so the discarded branch never holds inf (would poison jax.grad)
        sig_inv = jnp.where(sig_ok, 1.0 / jnp.where(sig_ok, sig_m, 1.0), 0.0)  # Sigma^dagger
        sol_m = (Vt_mm.T @ (sig_inv * (U_qm.T @ b0_q))) / cs_m
        X_gg = jnp.array([[sol_m[0], sol_m[1]], [sol_m[1], sol_m[2]]])
        c1_is = sol_m[3:15].reshape(2, 6); c2_is = sol_m[15:27].reshape(2, 6)
        res_tt = blocks(X_gg, c1_is, c2_is)
        Ustar_rel = jnp.linalg.norm(res_tt) / (jnp.linalg.norm(H_tt) + 1e-30)
        ev_min = jnp.linalg.eigvalsh(X_gg).min()               # SPD gate evaluated by caller
        G_gg = jnp.linalg.inv(X_gg)                            # Eq. (61) transverse-shear G

        # ---- second-order warping V2 (Eq. 64), for the Eq. (65)-(66) recovery only ----
        # Putting V = V0 eps + V1bar,a eps,a + V2 back into the energy and collecting the
        # V2-linear terms, everything pairing V2 with eps or eps,a cancels through the
        # lower-order Euler-Lagrange equations (+ the gauge), leaving the Eq.-(42) analog
        #     2 Pi2* = V2^T E V2 + 2 V2^T (Dbar21 e,11 + Dbar22 e,12 + Dbar23 e,22)
        # with the Eq. (43)-(44) drivers one rung up the ladder (V0 -> V1bar_a in the skew
        # part, D_lae -> D_lalb V0 in the direct part; the mixed driver gets both orders):
        #     Dbar21 = (D_hl1 - D_hl1^T) V11bar - D_l1l1 V0
        #     Dbar22 = (D_hl1 - D_hl1^T) V12bar + (D_hl2 - D_hl2^T) V11bar
        #              - (D_l1l2 + D_l1l2^T) V0
        #     Dbar23 = (D_hl2 - D_hl2^T) V12bar - D_l2l2 V0
        # Per the paper's Sec. 5 choice, the RELAXED first-order columns V1bar_a = V1a +
        # kernel c_a (Eq. 58) are the ones carried forward -- and here the constants
        # genuinely enter (D_hla @ kernel != 0).  V2's own energy content is O((h/l)^4),
        # below the model's resolution, so A6 and G are untouched; V2 exists purely to
        # recover the through-thickness fields at their leading order.
        #
        # c_a_is (2,6) padded with the zero w3 row -> c_a_ks (3,6); L_a_ns = kernel_nk @ c_a_ks
        c1_ks = jnp.zeros((3, 6)).at[:2].set(c1_is)
        c2_ks = jnp.zeros((3, 6)).at[:2].set(c2_is)
        V11bar_ns = V11_ns + jkernel_nk @ c1_ks                # V1bar columns (Eq. 58)
        V12bar_ns = V12_ns + jkernel_nk @ c2_ks
        D21bar_ns = (D_hl1_nn - D_hl1_nn.T) @ V11bar_ns - D_l1l1_nn @ V0_ns
        D22bar_ns = ((D_hl1_nn - D_hl1_nn.T) @ V12bar_ns + (D_hl2_nn - D_hl2_nn.T) @ V11bar_ns
                     - (D_l1l2_nn + D_l1l2_nn.T) @ V0_ns)
        D23bar_ns = (D_hl2_nn - D_hl2_nn.T) @ V12bar_ns - D_l2l2_nn @ V0_ns
        V2_n3s = solve_constrained(jnp.concatenate([D21bar_ns, D22bar_ns, D23bar_ns], axis=1))
        V21_ns = V2_n3s[:, :6]; V22_ns = V2_n3s[:, 6:12]; V23_ns = V2_n3s[:, 12:]  # Eq. (64)

        # NOTE on the LOAD COLUMNS (Yu Eqs. 29/45): V1L (E V1L + L = H psi psi^T L, one
        # more RHS on the same lu_piv) and the five V2L columns were prototyped and
        # validated here (faces exact by construction, sigma33 1.19% at S=10 on the
        # Pagano benchmark) and then removed by decision: the through-thickness
        # equilibrium integration of the recovered sigma13 delivers sigma33 with equal
        # or better accuracy for the pressure-loaded cases without extending the API,
        # and for beam-resultant loading (this code's primary use) the load drivers
        # vanish identically.  See git history (d1b52ac / 3e2f1d0) for the working
        # implementation and examples/benchmarks history for its validation.

        # returned in the PAPER symbols (the public dict keys carry no dimension suffixes)
        return (A6_ss, G_gg, X_gg, H_tt, Ustar_rel, V0_ns, V11_ns, V12_ns, c1_is, c2_is,
                ev_min, V11bar_ns, V12bar_ns, V21_ns, V22_ns, V23_ns)

    bk = dict(nlay=nlay, n_per_layer=n_per_layer, p=p, n_elem=n_elem, ndofs=ndofs,
              elem_layer=elem_layer,
              jit_single=jax.jit(single),
              jit_batch=jax.jit(jax.vmap(single, in_axes=(0, 0, None))))
    _BUCKETS[key] = bk
    return bk

def _node_grid(thick, n_per_layer, p, z_ref):
    """node_x exactly as the recovery expects (same origin as the reference plane)."""
    nlay = len(thick)
    layer_bot = np.concatenate([[0.0], np.cumsum(thick)])
    n_elem = nlay * n_per_layer
    node_x = np.empty(p * n_elem + 1)
    idx = 0
    for k in range(nlay):
        for s in range(n_per_layer):
            xl = layer_bot[k] + thick[k] * s / n_per_layer
            xr = layer_bot[k] + thick[k] * (s + 1) / n_per_layer
            for j in range(p):
                node_x[p * idx + j] = xl + (xr - xl) * j / p
            idx += 1
    node_x[p * n_elem] = layer_bot[-1]
    return node_x - z_ref


def _detilt_inplane(cols, node_x):
    """Project the TILT (x3-linear content) out of the IN-PLANE components of a
    warping-column block.

    Variables: cols (ndofs, 6) = nodal warping columns (dof order w1, w2, w3
    per node); node_x = the through-thickness node coordinates (reference-
    surface origin); W = (nnode, 3, 6) view; z2/m1 = trapezoid moments
    int x3^2 dx3 and int x3 w dx3 on the sorted grid.

    WHY (the theory, per Yu 2002 IJSS / Yu 2003 C&S): the warping gauge is
    ONLY <w_i> = 0 -- no first-moment constraint exists, deliberately: the
    x3-linear (tilt) content of the in-plane V1bar columns IS the transverse
    shear deformation (Yu constrains the triad normal to the deformed surface,
    so shear has exactly one home, the warping).  Yu's recovery is stated in
    CLASSICAL measures: when the 2-D solver is Reissner-like, its output R is
    converted by eps = R - D_alpha gamma,_alpha (Yu 2003 Eq. 50) BEFORE
    driving the warping.  This code's public API takes the RM measures R
    directly; using DETILTED columns in the Gamma_l (value) terms performs
    that conversion implicitly -- numerically equivalent to the literal
    Eq.-50 route in the asymptotic regime (0.043% vs 0.044% at S = 64 on
    Pagano caseA) and far better-behaved thick (26.8% vs 2373% at S = 4,
    where the eps-substitution itself is no longer asymptotic).  Callers must
    therefore pass R, NOT a pre-converted eps (that would double-correct).
    The DERIVATIVE use (Gamma_h, the transverse-shear recovery) keeps the RAW
    columns: there the tilt delivers the mean shear into the 3-D field
    (sigma_13 validated to 0.18% at S = 50).  Raw columns in a VALUE use next
    to z*phi would double-count the shear: caseA U1 85%/3.4% at S = 10/50 raw
    vs 2.1%/0.07% handled correctly; the in-plane second-order strain -76%
    raw vs +-2% detilted.  w3 keeps its tilt: U3 has no z*phi partner.
    """
    W = np.asarray(cols).reshape(len(node_x), 3, 6).copy()
    order = np.argsort(node_x)
    xs = np.asarray(node_x)[order]
    z2 = np.trapezoid(xs * xs, xs)
    for comp in (0, 1):
        m1 = np.trapezoid(xs[:, None] * W[order, comp, :], xs, axis=0)
        W[:, comp, :] -= np.outer(np.asarray(node_x), m1 / z2)
    return W.reshape(-1, 6)


def _pack(out, thick, angles_deg, C_layers, n_per_layer, p, fraction, elem_layer):
    (A6, G, X, H, Ustar_rel, V0, V11, V12, c1, c2, ev_min,
     V11b, V12b, V21, V22, V23) = [np.asarray(o) for o in out]
    thick = [float(t) for t in thick]
    spd = float(ev_min) > 0
    # the full RM plate law, Eqs. (40) + (61):  ABDG = [[A6, 0], [0, G]]
    # (rows/cols: e11, e22, g12, k11, k22, k12, 2g13, 2g23; None if X is not SPD)
    ABDG = None
    if spd:
        ABDG = np.zeros((8, 8))
        ABDG[:6, :6] = A6
        ABDG[6:, 6:] = G
    node_x = _node_grid(thick, n_per_layer, p, float(fraction) * sum(thick))
    return {"A6": A6, "G_msg": (G if spd else None), "ABDG": ABDG, "X": X, "H": H,
            "Ustar_rel": float(Ustar_rel),
            "V0": V0, "V11": V11, "V12": V12,
            "V11bar": V11b, "V12bar": V12b, "V21": V21, "V22": V22, "V23": V23,
            # detilted VALUE-use variants (Eq. 65 / the Gamma_l terms of Eq. 66;
            # see _detilt_inplane for why the raw columns double-count z*phi)
            "V11barD": _detilt_inplane(V11b, node_x),
            "V12barD": _detilt_inplane(V12b, node_x),
            "node_x": node_x,
            "elem_layer": elem_layer, "C_layers": list(C_layers), "elem_order": p,
            "angles": [float(a) for a in angles_deg], "c1": c1, "c2": c2}


def rm_plate_msg(thick, angles_deg, mat_names, material_db, n_per_layer=1, elem_order=4,
                 fraction=0.0):
    """Build the MSG-RM plate law for ONE laminate.  Returns a dict:
    A6 (6x6 ABD), G_msg (2x2, None if the fitted compliance is not SPD), ABDG (the full
    8x8 plate law [[A6,0],[0,G]], rows e11,e22,g12,k11,k22,k12,2g13,2g23), X (=G^{-1}),
    H (12x12), Ustar_rel (residual after projection / before), V0/V11/V12 (ndofs x 6),
    node_x, elem_layer, C_layers, elem_order, angles, c1, c2.

    Reference plane: ``fraction`` in [0, 1] of the total thickness -- 0 = OML (bottom,
    first ply) face (default), 0.5 = center, 1 = IML face.

    Discretization default = the paper's 5-noded (quartic, one element per ply) choice:
    with C constant per ply the exact warping is piecewise polynomial of degree
    V0: 2, V1: 3, V2: 4 (the paper's "piecewise, fourth-order polynomials" remark),
    so elem_order=4 represents the WHOLE ladder exactly -- measured sigma33 closure
    8.4e-5 with machine-zero face tractions vs 1.5e-2 for cubic subdivision, at ~3x
    fewer dofs; A6 and G are identical for any elem_order >= 3 (V1 cubic).

    jit-compiles once per (n_ply, n_per_layer, elem_order) bucket; batch many laminates
    with ``rm_plate_msg_batch`` for the vmapped fast path."""
    C_layers = np.array([rotated_stiffness_6x6(material_db[mat_names[k]]['E'],
                                               material_db[mat_names[k]]['G'],
                                               material_db[mat_names[k]]['nu'],
                                               angles_deg[k]) for k in range(len(thick))])
    bk = _bucket(len(thick), n_per_layer, elem_order)
    out = bk["jit_single"](jnp.asarray(np.asarray(thick, float)), jnp.asarray(C_layers),
                           jnp.asarray(float(fraction)))
    return _pack(out, thick, angles_deg, C_layers, n_per_layer, int(elem_order), fraction,
                 bk["elem_layer"])


def rm_plate_msg_batch(layups, material_db, n_per_layer=1, elem_order=4, fraction=0.0):
    """Vmapped fast path: MSG-RM plate law for MANY laminates at once.

    ``layups`` = list of dicts with keys mat_names / thick / angles (the ``layup_db``
    values of ``msg_mesh.load_yaml``).  Laminates are grouped into (n_ply) buckets --
    the hex/tet element-type-batch analog -- one jitted vmap call per bucket.
    Returns a list of ``rm_plate_msg`` dicts in input order."""
    groups = {}
    for i, l in enumerate(layups):
        groups.setdefault(len(l["thick"]), []).append(i)
    results = [None] * len(layups)
    for nlay, idxs in groups.items():
        thick_b = np.array([[float(t) for t in layups[i]["thick"]] for i in idxs])
        C_b = np.array([[rotated_stiffness_6x6(material_db[m]['E'], material_db[m]['G'],
                                               material_db[m]['nu'], float(a))
                         for m, a in zip(layups[i]["mat_names"], layups[i]["angles"])]
                        for i in idxs])
        bk = _bucket(nlay, n_per_layer, elem_order)
        outs = bk["jit_batch"](jnp.asarray(thick_b), jnp.asarray(C_b),
                               jnp.asarray(float(fraction)))
        for bi, i in enumerate(idxs):
            out_i = [o[bi] for o in outs]
            results[i] = _pack(out_i, layups[i]["thick"], layups[i]["angles"], C_b[bi],
                               n_per_layer, int(elem_order), fraction, bk["elem_layer"])
    return results


def _locate(obj, z):
    """element index + local operators at through-thickness x = z."""
    node_x = obj["node_x"]; p = obj["elem_order"]
    n_elem = len(obj["elem_layer"])
    e = int(np.clip(np.searchsorted(node_x[::p][1:], z, side="right"), 0, n_elem - 1))
    xl = node_x[p * e]; xr = node_x[p * e + p]; he = xr - xl
    xi = np.clip(2.0 * (z - xl) / he - 1.0, -1.0, 1.0)
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    dofs = np.arange(3 * p * e, 3 * p * e + 3 * (p + 1))
    return e, xi, he, nodes_xi, dofs, 0.5 * (xl + xr) + 0.5 * he * xi


def _warp_terms(obj, dofs, E6, dE1, dE2, dE11, dE12, dE22):
    """Eq. (58)+(64) local warping and the Gamma_l arguments of Eq. (66).

    w_loc = V0 e + V1bar,a e,a + V2ab e,ab       (the S(V0 + V1bar + V2) content)
    g_a   = (V0 e),a + (V1bar),a = V0 e,a + V11bar e,a1 + V12bar e,a2
    Uses the RELAXED first-order columns V11bar/V12bar (paper Sec. 5: carry V1bar,
    not V1, into the recovery); in Gamma_h the constants drop out anyway, in the
    Gamma_l terms (second order) they are genuine contributors.

    COLUMN VARIANTS BY USE: w_loc feeds Gamma_h (the through-thickness
    DERIVATIVE) and uses the RAW V1bar columns -- their tilt is what delivers
    the mean transverse shear into the sigma_a3 recovery (validated 0.18% at
    S = 50).  g_a feeds Gamma_l (the warping VALUE, the in-plane rows of the
    second-order recovery) and uses the DETILTED variants V11barD/V12barD: next
    to a plate solution the raw tilt double-counts z*phi (see _detilt_inplane;
    raw columns drove the caseA in-plane recovery to -76% at S = 10, detilted
    to +-2%)."""
    w_loc = (obj["V0"][dofs] @ E6 + obj["V11bar"][dofs] @ dE1 + obj["V12bar"][dofs] @ dE2
             + obj["V21"][dofs] @ dE11 + obj["V22"][dofs] @ dE12 + obj["V23"][dofs] @ dE22)
    g1 = obj["V0"][dofs] @ dE1 + obj["V11barD"][dofs] @ dE11 + obj["V12barD"][dofs] @ dE12
    g2 = obj["V0"][dofs] @ dE2 + obj["V11barD"][dofs] @ dE12 + obj["V12barD"][dofs] @ dE22
    return w_loc, g1, g2


def msgrm_strain_at_depth(obj, z, E6, dE1=None, dE2=None, dE11=None, dE12=None, dE22=None):
    """3-D Voigt strain at through-thickness x=z (same origin as node_x).

    With dE1/dE2 only: the FIRST-order recovery, Eq. (63).  Passing the second gradients
    dE11/dE12/dE22 (= E6,11 / E6,12 / E6,22) activates the SECOND-order recovery, Eq. (66):
        Gam = Gamma_h S(V0+V1bar+V2) + Gamma_eps eps
              + Gamma_l1 S(V0,1 + V1bar,1) + Gamma_l2 S(V0,2 + V1bar,2)
    which is what carries the through-thickness components at their leading order.
    For a SURFACE-PRESSURE-loaded plate, sigma33 is obtained by through-thickness
    equilibrium integration of the recovered sigma13/sigma23 amplitudes (see the
    load-column note in the homogenization kernel).  Returns (Gam6, Sig6,
    ply_angle_deg)."""
    zeros = np.zeros(6)
    dE1 = zeros if dE1 is None else np.asarray(dE1, float)
    dE2 = zeros if dE2 is None else np.asarray(dE2, float)
    dE11 = zeros if dE11 is None else np.asarray(dE11, float)
    dE12 = zeros if dE12 is None else np.asarray(dE12, float)
    dE22 = zeros if dE22 is None else np.asarray(dE22, float)
    e, xi, he, nodes_xi, dofs, x_q = _locate(obj, z)
    Gamma_h = _plate_B(nodes_xi, xi, he)
    Gamma_l1, Gamma_l2 = _grad_ops(nodes_xi, xi)
    Gamma_eps = _E0 + x_q * _E1
    w_loc, g1, g2 = _warp_terms(obj, dofs, E6, dE1, dE2, dE11, dE12, dE22)
    Gam = Gamma_h @ w_loc + Gamma_eps @ E6 + Gamma_l1 @ g1 + Gamma_l2 @ g2
    k = obj["elem_layer"][e]
    Sig = np.asarray(obj["C_layers"][k]) @ Gam
    return Gam, Sig, obj["angles"][k]


def msgrm_warping_at_depth(obj, z, E6, dE1=None, dE2=None, dE11=None, dE12=None, dE22=None):
    """The 3-D warping displacement S(V0 + V1bar + V2) at x=z -- the SG part of the
    Eq. (65) displacement recovery (u_2d and the x3-rotation term are the plate
    solution's contribution and are added by the caller).  Returns w (3,).

    COMPOSITION RULE (settled by the controlled caseA sweep, S = 4..64): use
    the RAW columns here and compose with the KIRCHHOFF z-linear term,
        U_alpha = u_alpha^2d - x3 w,alpha + w_alpha ,   U3 = w + w3 ,
    NOT with x3*phi_alpha.  The raw V1bar tilt is exactly the mean-shear
    content the Kirchhoff term lacks (U1 error 1.41%/0.016% at S = 10/64;
    composing with x3*phi double-counts it: 85%/2.1% raw/detilted)."""
    zeros = np.zeros(6)
    dE1 = zeros if dE1 is None else np.asarray(dE1, float)
    dE2 = zeros if dE2 is None else np.asarray(dE2, float)
    dE11 = zeros if dE11 is None else np.asarray(dE11, float)
    dE12 = zeros if dE12 is None else np.asarray(dE12, float)
    dE22 = zeros if dE22 is None else np.asarray(dE22, float)
    e, xi, he, nodes_xi, dofs, x_q = _locate(obj, z)
    w_loc = (obj["V0"][dofs] @ E6
             + obj["V11bar"][dofs] @ dE1 + obj["V12bar"][dofs] @ dE2
             + obj["V21"][dofs] @ dE11 + obj["V22"][dofs] @ dE12
             + obj["V23"][dofs] @ dE22)
    N = _lagrange_N(nodes_xi, xi)
    w_nodes = w_loc.reshape(-1, 3)                    # (p+1, 3) nodal warping in the element
    return N @ w_nodes


