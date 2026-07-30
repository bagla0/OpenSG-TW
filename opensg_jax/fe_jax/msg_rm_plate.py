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
                   (kernel^T D_he = 0); returned as ``lam0_check`` (~1e-15)
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

Validation:  python -m opensg_jax.fe_jax.msg_rm_plate
homogeneous isotropic -> G_msg = 5/6 G h (nu=0);  orthotropic laminates -> G_msg ~=
Whitney/complementary-energy transverse_shear_stiffness; A6 == compute_ABD_matrix.
"""
import numpy as np
import jax
import jax.numpy as jnp

from jax.scipy.linalg import lu_factor, lu_solve

from .msg_materials import rotated_stiffness_6x6, _plate_B, compute_ABD_matrix
from .msg_transverse_shear import transverse_shear_stiffness


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
    kernel = np.zeros((ndofs, 3))
    kernel[0::3, 0] = 1.0; kernel[1::3, 1] = 1.0; kernel[2::3, 2] = 1.0
    nodes_of_e = np.stack([np.arange(p * e, p * e + p + 1) for e in range(n_elem)])
    S_int_ref = np.array([sum(w_g[q] * _lagrange_N(nodes_xi, xi_g[q])[a] for q in range(len(xi_g)))
                          for a in range(p + 1)])   # int N_a dxi over [-1,1] (for H = <S^T S>)

    # D_1, D_2 (Eq. 51): Boolean selectors of eps = R - D_a gamma,_a, code strain order
    D1 = np.zeros((6, 2)); D2 = np.zeros((6, 2))
    D1[3, 0] = 1.0; D1[5, 1] = 1.0        # k11 <- 2g13,1 ; k12 <- 2g23,1
    D2[4, 1] = 1.0; D2[5, 0] = 1.0        # k22 <- 2g23,2 ; k12 <- 2g13,2

    # unit directions of the 27 LS unknowns [x11,x12,x22, c1(12), c2(12)] (Eq. 58: 3 + 24)
    units = []
    for j in range(27):
        pj = np.zeros(27); pj[j] = 1.0
        units.append((np.array([[pj[0], pj[1]], [pj[1], pj[2]]]),
                      pj[3:15].reshape(2, 6), pj[15:27].reshape(2, 6)))

    jGh = jnp.asarray(Gh_ref); jGl1 = jnp.asarray(Gl1_ref); jGl2 = jnp.asarray(Gl2_ref)
    jE0 = jnp.asarray(_E0); jE1 = jnp.asarray(_E1)
    jdofs = jnp.asarray(dofs_e); jlayer = jnp.asarray(elem_layer); jsub = jnp.asarray(sub_of_elem)
    jkernel = jnp.asarray(kernel)
    jnodes_e = jnp.asarray(nodes_of_e); jS_int = jnp.asarray(S_int_ref)
    jD1 = jnp.asarray(D1); jD2 = jnp.asarray(D2)
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
                Gamma_h = (2.0 / he_e) * jGh[q]                 # [Gamma_h S]: d/dx3 strains
                Gamma_l1 = jGl1[q]; Gamma_l2 = jGl2[q]          # [Gamma_l1 S], [Gamma_l2 S]
                x_q = xl_e + he_e * 0.5 * (1.0 + jxi_g[q])
                Gamma_eps = jE0 + x_q * jE1                     # Gamma_eps: e + x3*k rows
                D_hh_ll += Gamma_h.T @ C_e @ Gamma_h * dw
                D_he_ls += Gamma_h.T @ C_e @ Gamma_eps * dw
                D_ee_ss += Gamma_eps.T @ C_e @ Gamma_eps * dw
                D_hl1_ll += Gamma_h.T @ C_e @ Gamma_l1 * dw
                D_hl2_ll += Gamma_h.T @ C_e @ Gamma_l2 * dw
                D_l1l1_ll += Gamma_l1.T @ C_e @ Gamma_l1 * dw
                D_l1l2_ll += Gamma_l1.T @ C_e @ Gamma_l2 * dw
                D_l2l2_ll += Gamma_l2.T @ C_e @ Gamma_l2 * dw
                D_l1e_ls += Gamma_l1.T @ C_e @ Gamma_eps * dw
                D_l2e_ls += Gamma_l2.T @ C_e @ Gamma_eps * dw
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

        # assembled SG matrices, paper Eq. 30 names (suffix = dims: n = warping dofs, s = 6 plate strains)
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
        w_dof = jnp.repeat(w_node, 3)                          # <.> weights on the dofs
        psi = jkernel / jnp.sqrt(h)                            # Eq. (31): psi^T H psi = I3
        Hpsi = w_dof[:, None] * psi                            # H @ psi  (ndofs, 3)
        # Eq. (34) solved DIRECTLY as the Lagrange-multiplier saddle-point system:
        #   [[D_hh, H psi], [psi^T H, 0]] [V; Lam] = [-rhs; 0]
        # the multiplier row reproduces Eq. (35) automatically (Lam = -psi^T rhs: zero at
        # zeroth order since Gamma_h of a constant vanishes, = -S_a/sqrt(h) at first
        # order), and the constraint row enforces <w_i> = 0 exactly, so Eq. (39) is
        # built into the solve.  scl balances the two blocks for conditioning only.
        scl = jnp.max(jnp.abs(jnp.diag(D_hh_nn)))
        KKT = jnp.block([[D_hh_nn, scl * Hpsi],
                         [scl * Hpsi.T, jnp.zeros((3, 3))]])
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
        V0 = solve_constrained(D_he_ns)
        A6 = D_ee_ss + V0.T @ D_he_ns
        lam0_rel = (jnp.max(jnp.abs(jkernel.T @ D_he_ns))
                    / (jnp.max(jnp.abs(D_he_ns)) + 1e-300))    # theory: 0 (roundoff ~n*eps)

        # first-order drivers, Eqs. (43)-(44):  D_a = (D_hla - D_hla^T) V0_hat - D_lae
        #   named D1bar/D2bar because Eq. (51) reuses the symbols D_1, D_2 for the Boolean
        #   strain selectors (the paper overloads them; see D1/D2 in the RM projection below)
        #   D_lah = D_hla^T exactly (same quadrature, symmetric C), so no separate assembly
        D1bar = (D_hl1_nn - D_hl1_nn.T) @ V0 - D_l1e_ns
        D2bar = (D_hl2_nn - D_hl2_nn.T) @ V0 - D_l2e_ns

        # first-order warping columns V11, V12 (paper Eq. 45).
        #
        # The paper gives the Euler-Lagrange equation of the ZEROTH-order functional
        # (Eq. 33) as Eq. (34), but never writes the one for the FIRST-order functional
        # Eq. (42),  2*Pi*_1 = V1^T E V1 + 2 V1^T Dbar_1 eps,1 + 2 V1^T Dbar_2 eps,2
        # + 2 V1^T L.  Varying V1 subject to Eq. (31) with multiplier Lambda_1 gives
        #
        #     E V1 + Dbar_1 eps,1 + Dbar_2 eps,2 + L = H psi Lambda_1          (E-L of 42)
        #
        # and premultiplying by psi^T, using E psi = 0 (psi spans null(E)) and
        # psi^T H psi = I (Eq. 31), the multiplier follows as the analog of Eq. (35):
        #
        #     Lambda_1 = psi^T (Dbar_1 eps,1 + Dbar_2 eps,2 + L)
        #
        # Substituting Eq. (45), V1 = V11 eps,1 + V12 eps,2 + V1L, and collecting the
        # coefficients of the ARBITRARY, independent 6-vectors eps,1 and eps,2 factors
        # eps,a out and splits the single vector equation into MATRIX equations
        #
        #     E V1a + Dbar_a = H psi (psi^T Dbar_a),  a = 1,2   [V1a, Dbar_a are (ndofs x 6)]
        #
        # -- i.e. 12 linear systems: 6 unit plate-strain-gradient load cases (one per
        # component of eps,a) for each in-plane direction a.  All 12 share the operator
        # of Eq. (34), so they are ONE LU factorization vmapped over the 12 columns.
        # The Eq. (39) gauge is enforced by the KKT constraint row, not applied after.
        # (V1L omitted: this module carries no applied loads, L = 0.)
        # Unlike the zeroth order, psi^T Dbar_a != 0 here (Gamma_l of a constant is nonzero),
        # so this multiplier genuinely acts -- and its content, kernel^T Dbar_a, is exactly
        # the S_a driving the 24 relaxed constants below.
        V1 = solve_constrained(jnp.concatenate([D1bar, D2bar], axis=1))    # all 12 at once
        V11 = V1[:, :6]; V12 = V1[:, 6:]

        # gradient energy blocks B, C, D of Eq. (47) -> H = [[B,C],[C^T,D]] over [E,1; E,2]
        H11 = V0.T @ D_l1l1_nn @ V0 + D1bar.T @ V11
        H12 = V0.T @ D_l1l2_nn @ V0 + 0.5 * (D1bar.T @ V12 + V11.T @ D2bar)
        H22 = V0.T @ D_l2l2_nn @ V0 + D2bar.T @ V12
        # B and D are ANALYTICALLY symmetric: under the Eq. (31) constraint the E-L equation
        # gives V11^T Dbar_1 = -V11^T E V11, and E = E^T.  So this only removes roundoff
        # (measured 5e-18..5e-12 relative) -- but it matters, because every one of the 27 LS
        # unknowns enters the DIAGONAL blocks symmetrically (X via AD X AD^T with X = X^T,
        # the constants via c^T S + S^T c).  An antisymmetric part is therefore unreachable
        # by the fit and would sit in the residual as an irreducible Ustar_rel floor.
        # C (= H12) is genuinely NOT symmetric (asymmetry of order 1) and must NOT be
        # symmetrized; its transpose placement in H below is what handles it.
        H11 = 0.5 * (H11 + H11.T); H22 = 0.5 * (H22 + H22.T)
        H = jnp.block([[H11, H12], [H12.T, H22]])

        # ---- RM projection (Yu 2003 sec. 4): E = R - D1 g,1 - D2 g,2 ; equilibrium swap
        #      Eq. (54); LS over X = G^{-1} (sym 2x2) and relaxed constants c1,c2 (2x6 each) ----
        S1 = (jkernel.T @ D1bar)[:2]           # (2,6) in-plane constant shifts (Yu's 24; the
        S2 = (jkernel.T @ D2bar)[:2]           #  w3 row of kernel^T D_abar = 0, monoclinic)
        AD1 = A6 @ jD1; AD2 = A6 @ jD2

        def blocks(X, c1, c2):
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
            Bs = H11 + AD1 @ X @ AD1.T + c1.T @ S1 + S1.T @ c1
            Cs = H12 + AD1 @ X @ AD2.T + c1.T @ S2 + S1.T @ c2
            Ds = H22 + AD2 @ X @ AD2.T + c2.T @ S2 + S2.T @ c2
            return jnp.block([[Bs, Cs], [Cs.T, Ds]])

        # linear LS ("78 equations, 27 unknowns", text after Eq. 57): columns by unit probing
        M0 = blocks(jnp.zeros((2, 2)), jnp.zeros((2, 6)), jnp.zeros((2, 6)))   # == H exactly
        b0 = -M0.ravel()
        Amat = jnp.stack([blocks(*unit).ravel() + b0 for unit in junits], axis=1)
        # Column equilibration: the X-columns scale like A^2 (~1e13) and the c-columns like
        # S (~1e9), a ~1e4 spread that pushes cond(Amat) to ~5e19.  Scaling each column to
        # unit norm brings it to ~8e15.  Only the redundant constants move; X is invariant.
        cs = jnp.linalg.norm(Amat, axis=0)
        cs = jnp.where(cs == 0, 1.0, cs)

        # --- truncated-SVD MINIMUM-NORM solution ------------------------------------------
        # Ascher & Greif, "A First Course in Numerical Methods" (SIAM 2011), Sec. 8.2 p.235:
        #     Amat = U Sigma V^T,   x = V Sigma^dagger U^T b0,
        #     Sigma^dagger_ii = 1/sigma_i  if sigma_i != 0,  else 0
        # with "!= 0" realised as sigma_i > _LS_RCOND * sigma_max.
        #
        # Why this route and not Ch.6's: Amat is 144x27 but RANK-DEFICIENT (rank 26 -- one
        # direction, antisymmetric mixing of the c's, changes nothing), so Ch.6's full-column-
        # rank assumption fails.  Measured on a [45/0/-30/90] laminate: this route is exact to
        # 6e-16, normal equations + Cholesky 2e-14, and QR back-substitution only 7e-5 (it
        # divides by a ~1e-13 diagonal of R).  The dropped direction has zero X-component, so
        # G is unique regardless; min-norm merely pins the arbitrary relaxation constants.
        U_ls, sig, Vt_ls = jnp.linalg.svd(Amat / cs, full_matrices=False)
        sig_ok = sig > _LS_RCOND * sig[0]
        # guard the reciprocal so the discarded branch never holds inf (would poison jax.grad)
        sig_inv = jnp.where(sig_ok, 1.0 / jnp.where(sig_ok, sig, 1.0), 0.0)   # Sigma^dagger
        sol = (Vt_ls.T @ (sig_inv * (U_ls.T @ b0))) / cs
        X = jnp.array([[sol[0], sol[1]], [sol[1], sol[2]]])
        c1 = sol[3:15].reshape(2, 6); c2 = sol[15:27].reshape(2, 6)
        res = blocks(X, c1, c2)
        Ustar_rel = jnp.linalg.norm(res) / (jnp.linalg.norm(H) + 1e-30)
        ev_min = jnp.linalg.eigvalsh(X).min()                  # SPD gate evaluated by caller
        G = jnp.linalg.inv(X)                                  # Eq. (61) transverse-shear G

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
        # LOAD CHAIN (theory, deliberately not implemented): with surface tractions the
        # paper adds L = S+^T tau - S-^T beta - <S^T phi> (Eq. 29) and the ladder gains
        # load columns V1L (Eq. 45) and V2L, plus the load-stiffness terms F and P of
        # Eq. (47) ("thermal-like" forcing of the 2-D solver, Eq. 61).  Numerically each
        # is ONE more RHS against the same lu_piv.  They are omitted here because
        # (i) A6 and G are load-independent (L never enters B/C/D, hence not U*), and
        # (ii) in the blade pipeline loads enter through the beam/plate solution (FF),
        # so the recovery below carries the SELF-EQUILIBRATED part of sigma33; the
        # surface-pressure-carrying part (sigma33 ramp from -q to 0 through the wall)
        # would need V2L and scales with the local panel pressure q -- negligible next
        # to the interlaminar signals recovered here.
        c1f = jnp.zeros((3, 6)).at[:2].set(c1)
        c2f = jnp.zeros((3, 6)).at[:2].set(c2)
        V11b = V11 + jkernel @ c1f                             # V1bar columns (Eq. 58)
        V12b = V12 + jkernel @ c2f
        D21bar = (D_hl1_nn - D_hl1_nn.T) @ V11b - D_l1l1_nn @ V0
        D22bar = ((D_hl1_nn - D_hl1_nn.T) @ V12b + (D_hl2_nn - D_hl2_nn.T) @ V11b
                  - (D_l1l2_nn + D_l1l2_nn.T) @ V0)
        D23bar = (D_hl2_nn - D_hl2_nn.T) @ V12b - D_l2l2_nn @ V0
        V2 = solve_constrained(jnp.concatenate([D21bar, D22bar, D23bar], axis=1))
        V21 = V2[:, :6]; V22 = V2[:, 6:12]; V23 = V2[:, 12:]  # Eq. (64) column blocks

        return (A6, G, X, H, Ustar_rel, V0, V11, V12, c1, c2, ev_min, lam0_rel,
                V11b, V12b, V21, V22, V23)

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


def _pack(out, thick, angles_deg, C_layers, n_per_layer, p, fraction, elem_layer):
    (A6, G, X, H, Ustar_rel, V0, V11, V12, c1, c2, ev_min, lam0_rel,
     V11b, V12b, V21, V22, V23) = [np.asarray(o) for o in out]
    thick = [float(t) for t in thick]
    return {"A6": A6, "G_msg": (G if float(ev_min) > 0 else None), "X": X, "H": H,
            "Ustar_rel": float(Ustar_rel), "lam0_check": float(lam0_rel),
            "V0": V0, "V11": V11, "V12": V12,
            "V11bar": V11b, "V12bar": V12b, "V21": V21, "V22": V22, "V23": V23,
            "node_x": _node_grid(thick, n_per_layer, p, float(fraction) * sum(thick)),
            "elem_layer": elem_layer, "C_layers": list(C_layers), "elem_order": p,
            "angles": [float(a) for a in angles_deg], "c1": c1, "c2": c2}


def rm_plate_msg(thick, angles_deg, mat_names, material_db, n_per_layer=1, elem_order=4,
                 fraction=0.0):
    """Build the MSG-RM plate law for ONE laminate.  Returns a dict:
    A6 (6x6 ABD), G_msg (2x2, None if the fitted compliance is not SPD), X (=G^{-1}),
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
    Gamma_l terms (second order) they are genuine contributors."""
    w_loc = (obj["V0"][dofs] @ E6 + obj["V11bar"][dofs] @ dE1 + obj["V12bar"][dofs] @ dE2
             + obj["V21"][dofs] @ dE11 + obj["V22"][dofs] @ dE12 + obj["V23"][dofs] @ dE22)
    g1 = obj["V0"][dofs] @ dE1 + obj["V11bar"][dofs] @ dE11 + obj["V12bar"][dofs] @ dE12
    g2 = obj["V0"][dofs] @ dE2 + obj["V11bar"][dofs] @ dE12 + obj["V12bar"][dofs] @ dE22
    return w_loc, g1, g2


def msgrm_strain_at_depth(obj, z, E6, dE1=None, dE2=None, dE11=None, dE12=None, dE22=None):
    """3-D Voigt strain at through-thickness x=z (same origin as node_x).

    With dE1/dE2 only: the FIRST-order recovery, Eq. (63).  Passing the second gradients
    dE11/dE12/dE22 (= E6,11 / E6,12 / E6,22) activates the SECOND-order recovery, Eq. (66):
        Gam = Gamma_h S(V0+V1bar+V2) + Gamma_eps eps
              + Gamma_l1 S(V0,1 + V1bar,1) + Gamma_l2 S(V0,2 + V1bar,2)
    which is what carries the through-thickness components (sigma33 in particular) at
    their leading order.  Returns (Gam6, Sig6, ply_angle_deg)."""
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
    solution's contribution and are added by the caller).  Returns w (3,)."""
    zeros = np.zeros(6)
    dE1 = zeros if dE1 is None else np.asarray(dE1, float)
    dE2 = zeros if dE2 is None else np.asarray(dE2, float)
    dE11 = zeros if dE11 is None else np.asarray(dE11, float)
    dE12 = zeros if dE12 is None else np.asarray(dE12, float)
    dE22 = zeros if dE22 is None else np.asarray(dE22, float)
    e, xi, he, nodes_xi, dofs, x_q = _locate(obj, z)
    w_loc, _, _ = _warp_terms(obj, dofs, E6, dE1, dE2, dE11, dE12, dE22)
    N = _lagrange_N(nodes_xi, xi)
    w_nodes = w_loc.reshape(-1, 3)                    # (p+1, 3) nodal warping in the element
    return N @ w_nodes


if __name__ == "__main__":
    # ---- validation 1: homogeneous isotropic -> G = 5/6 G h ----
    mdb = {"iso": {"E": [70e9] * 3, "G": [70e9 / 2.6] * 3, "nu": [0.3] * 3, "rho": 1.0}}
    h = 0.01
    r = rm_plate_msg([h], [0.0], ["iso"], mdb, n_per_layer=4, fraction=0.5)
    Gh = 70e9 / 2.6 * h
    print("iso: G_msg/(Gh) diag =", None if r["G_msg"] is None else np.diag(r["G_msg"]) / Gh,
          " target 5/6 = %.6f   Ustar_rel %.3e" % (5.0 / 6.0, r["Ustar_rel"]))
    Gw = transverse_shear_stiffness([h], [0.0], ["iso"], mdb)[0]
    print("     Whitney diag/(Gh) =", np.diag(Gw) / Gh)
    print("     zeroth-order multiplier |kernel^T D_he|/|D_he| = %.2e  (theory: 0, RHS _|_ kernel)"
          % r["lam0_check"])
    mdb0 = {"iso0": {"E": [70e9] * 3, "G": [35e9] * 3, "nu": [0.0] * 3, "rho": 1.0}}
    r0 = rm_plate_msg([h], [0.0], ["iso0"], mdb0, n_per_layer=4, fraction=0.5)
    print("     nu=0 : G_msg/(Gh) =", None if r0["G_msg"] is None else np.diag(r0["G_msg"]) / (35e9 * h),
          " Ustar %.2e" % r0["Ustar_rel"])
    rf = rm_plate_msg([h], [0.0], ["iso"], mdb, n_per_layer=12, elem_order=3, fraction=0.5)
    print("     fine : G_msg/(Gh) =", None if rf["G_msg"] is None else np.diag(rf["G_msg"]) / Gh)
    A_ref = compute_ABD_matrix([h], [0.0], ["iso"], mdb, n_per_layer=4, z_ref=h / 2)[0]
    print("     |A6 - compute_ABD(z_ref=h/2)| =", np.max(np.abs(r["A6"] - np.asarray(A_ref)[:6, :6])))

    # ---- validation 2: [0/90/0] Pagano-style graphite/epoxy ----
    mdb2 = {"gr": {"E": [172.4e9, 6.89e9, 6.89e9], "G": [3.45e9, 1.38e9, 3.45e9],
                   "nu": [0.25, 0.25, 0.25], "rho": 1.0}}
    thk = [0.005, 0.005, 0.005]; ang = [0.0, 90.0, 0.0]; mats = ["gr"] * 3
    r2 = rm_plate_msg(thk, ang, mats, mdb2, n_per_layer=4, fraction=0.5)
    Gw2 = transverse_shear_stiffness(thk, ang, mats, mdb2)[0]
    print("[0/90/0]: G_msg =", None if r2["G_msg"] is None else np.array2string(r2["G_msg"], precision=4))
    print("          Whitney=", np.array2string(Gw2, precision=4), "  Ustar_rel %.3e" % r2["Ustar_rel"])

    # ---- validation 3: web sandwich biax/foam/biax (s10 materials) ----
    mdb3 = {"biax": {"E": [11.5e9, 11.5e9, 1.3e10], "G": [11.8e9, 3.5e9, 3.5e9],
                     "nu": [0.5, 0.09, 0.09], "rho": 1.0},
            "foam": {"E": [1.42e8] * 3, "G": [6.0e7] * 3, "nu": [0.2] * 3, "rho": 1.0}}
    thk = [0.002, 0.042, 0.002]; ang = [0.0] * 3; mats = ["biax", "foam", "biax"]
    r3 = rm_plate_msg(thk, ang, mats, mdb3, n_per_layer=4, fraction=0.5)
    Gw3 = transverse_shear_stiffness(thk, ang, mats, mdb3)[0]
    print("web sandwich: G_msg =", None if r3["G_msg"] is None else np.array2string(r3["G_msg"], precision=4))
    print("              Whitney=", np.array2string(Gw3, precision=4), "  Ustar_rel %.3e" % r3["Ustar_rel"])

    # ---- validation 4: batch API == single-call API ----
    lay_batch = [{"mat_names": mats, "thick": thk, "angles": ang},
                 {"mat_names": ["gr"] * 3, "thick": [0.005] * 3, "angles": [0.0, 90.0, 0.0]}]
    rb = rm_plate_msg_batch(lay_batch, {**mdb2, **mdb3}, fraction=0.5)
    print("batch == single:",
          np.max(np.abs(rb[0]["G_msg"] - r3["G_msg"])) / np.max(np.abs(r3["G_msg"])) < 1e-8 and
          np.max(np.abs(rb[1]["G_msg"] - r2["G_msg"])) / np.max(np.abs(r2["G_msg"])) < 1e-8)
