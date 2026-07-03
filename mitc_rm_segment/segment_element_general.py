"""
segment_element_general.py -- GENERAL 3D-taper Reissner-Mindlin surface operators
==================================================================================
Implements the GENERAL RM shell strains and curvature strains of the OpenSG-RM
paper (Overleaf "OpenSG: RM to Timoshenko beam", Shell Strains section + Appendix
A.3-A.8), of which the prismatic cross-section operator (segment_element._quad_ops
/ msg_rm_timo._elem_BD_BG_BL) is the x_{1;1}=1, x_{1;2}=0 SIMPLIFIED form.

NOTATION (paper -> code)
------------------------
  zeta_1, zeta_2   surface coordinates along the in-plane axial tangent a1 and
                   hoop tangent a2 (unit, orthonormal).  d/dzeta_alpha of the
                   shape functions: D1a, D2a  (from the 2x2 metric).
  x_{i;alpha}      direction cosines  = i-th GLOBAL component of a_alpha, with
                   the global index i mapped to (1=beam axis, 2,3=cross coords):
                     x11=a1.b1  x21=a1.b2  x31=a1.b3     (b = beam frame)
                     x12=a2.b1  x22=a2.b2  x32=a2.b3
  y_i = C_{3i}     normal direction cosines: y1=n.b1, y2=n.b2, y3=n.b3, with
                   n = a1 x a2 (right-handed; inward for a CCW hoop).  In the
                   prismatic limit  (y2,y3) = (-xdot3, +xdot2)  so  C33 -> xdot2,
                   reproducing the paper's 1/(2 xdot2) drilling factors.
  w_{i|alpha}      SG-surface derivative of the fluctuation w_i along zeta_alpha
  w'_i, omega'_b   x1-(macro)-derivative fields (the Gamma_l argument)
  kappa_{1i}       beam curvatures (kappa_1 twist, kappa_2, kappa_3 bending);
                   beam strain vector eb = [gamma_11, k1, k2, k3]
  Rn1, Rn2         swept-area measures (A.4):
                     Rn1 = x2*x_{3;1} - x3*x_{2;1},  Rn2 = x2*x_{3;2} - x3*x_{2;2}

DOF per node: [w1, w2, w3, omega_1, omega_2]  (w_i global; omega_beta = the two
RM shear rotations, beta a GLOBAL index 1,2 as in A.3/A.8 x_{beta;alpha} pairing).

DRILLING (A.3, A.5, A.6):
  omega_3   = S/(2 C33) - (C3b/C33) omega_b                              (A.3)
      S     = k1 (x11 Rn2 - x12 Rn1) + w'_i (x11 x_{i;2} - x12 x_{i;1})
              + (w_{i|1} x_{i;2} - w_{i|2} x_{i;1})
  omega'_3  = (w'_{i|1} x_{i;2} - w'_{i|2} x_{i;1})/(2 C33) - (C3b/C33) omega'_b
              [ kappa'_1 and w'' terms neglected, order eps^3 -- A.5 ]
  Lambda_a  = omega'_3 x_{1;a} + omega_{3|a}                             (A.6)
  omega_{3|a}: differentiate the omega_3 OPERATOR along zeta_a:
      * shape-function part: KEPT where it stays first-order
        (the -(C3b/C33) omega_{b|a} term);  the w_{i|1|a}, w_{i|2|a} pieces are
        SECOND SG derivatives -- NOT RESOLVABLE in C0 RM -> DROPPED (this is the
        established RM decision: no second derivative of the fluctuations).
      * coefficient part: contour (zeta_2) derivatives of the direction cosines
        via the initial curvature k22 of the wall (Frenet along the hoop):
            (x_{i;2})_{;2} = k22 * y_i ,   (y_i)_{;2} = -k22 * x_{i;2} ,
            (x_{i;1})_{;2} = 0 (+geodesic terms, dropped on flat-ish quads),
            (x_i)_{;a}     = x_{i;a}      (coordinates),
        and all (.)_{;1} coefficient derivatives (taper rate of the frame) are
        higher-order -> dropped, marked below.

CURVATURE STRAINS (A.8), with Lambda as above:
  k^s_11      =  x11 x_{i;2} k_{1i}  +  x_{b;2} x11 omega'_b
               + x_{b;2} omega_{b|1} +  x_{3;2} Lambda_1
  k^s_22      = -[ x12 x_{i;1} k_{1i} + x_{b;1} x12 omega'_b
               + x_{b;1} omega_{b|2} + x_{3;1} Lambda_2 ]
  k^s_12+21   =  k_{1i}(x12 x_{i;2} - x11 x_{i;1})
               + omega'_b (x12 x_{b;2} - x11 x_{b;1})
               + (x_{b;2} omega_{b|2} - x_{b;1} omega_{b|1})
               + x_{3;2} Lambda_2 - x_{3;1} Lambda_1

MEMBRANE STRAINS (paper, Shell Strains):
  e_11   = x11^2 g11 + x11(x2 x31 - x3 x21) k1 + x11^2 (x3 k2 - x2 k3)
           + x_{i;1} w_{i|1} + x11 x_{i;1} w'_i
  e_22   = x12^2 g11 + x12(x2 x32 - x3 x22) k1 + x12^2 (x3 k2 - x2 k3)
           + x_{i;2} w_{i|2} + x12 x_{i;2} w'_i
  2e_12  = 2 x11 x12 g11 + [x2(x11 x32 + x12 x31) - x3(x11 x22 + x12 x21)] k1
           + 2 x11 x12 (x3 k2 - x2 k3)
           + (x_{i;1} w_{i|2} + x_{i;2} w_{i|1}) + (x11 x_{i;2} + x12 x_{i;1}) w'_i

TRANSVERSE SHEAR (prismatic-consistent minimal generalization; the fully
drilling-boosted general gamma_13 (eq12 form) is 1-D-validated and marked TODO
for the surface element -- it changes only GA2/GA3, not the taper diagonals):
  2g_13  = y_i w_{i|1} + omega_2 + x11 y_i w'_i
  2g_23  = y_i w_{i|2} - omega_1

PRISMATIC REDUCTION CHECK: x11=1, x12=0, x21=x31=0, D1->0, (y2,y3)=(-xd3,xd2)
reproduces eq:prism of the paper exactly (incl. omega'_2/xd2 and the
(xd3/2xd2)-terms through Lambda) minus the dropped second-derivative pieces.
"""
import numpy as np
from segment_element import _bilinear, dirichlet_solve, compute_k22, build_C_Psi_segment  # noqa

# beam-strain order [g11, k1, k2, k3]; DOF order per node [w1,w2,w3,om1,om2]
NDOF = 5
OM3_SIGN = -1.0   # sign convention of omega_3 (A.3); prismatic identity decides


def _surf_frame(X, e3_mat, xi, eta, cross, ax):
    """Geometric surface frame + in-plane derivative operator at (xi, eta).

    a2 = hoop tangent (J_xi direction), a1 = in-plane AXIAL tangent (J_eta
    Gram-Schmidt'ed against a2 -- TILTED by the taper, this is where x21,x31
    come from), n = a1 x a2 sign-matched to the stored material normal e3_mat.
    Returns (N, D1, D2, dA, cosines dict).
    """
    N, dNx, dNe = _bilinear(xi, eta)
    Jxi = dNx @ X; Jeta = dNe @ X
    a2 = Jxi / np.linalg.norm(Jxi)
    a1 = Jeta - (Jeta @ a2) * a2
    a1 = a1 / np.linalg.norm(a1)
    n = np.cross(a1, a2)
    if n @ e3_mat < 0.0:                      # keep the mesh's inward/outward choice
        n = -n; a1 = -a1                      # flip a1 too -> right-handed (a1,a2,n)
    # in-plane derivatives: [d/dz1; d/dz2] = (G^T)^-1 [d/dxi; d/deta]
    G = np.array([[a1 @ Jxi, a1 @ Jeta], [a2 @ Jxi, a2 @ Jeta]])
    d = (np.linalg.inv(G.T) @ np.vstack([dNx, dNe]))      # (2,4): rows = [D1; D2]
    dA = np.linalg.norm(np.cross(Jxi, Jeta))
    x = N @ X
    c = dict(
        x11=a1[ax], x21=a1[cross[0]], x31=a1[cross[1]],
        x12=a2[ax], x22=a2[cross[0]], x32=a2[cross[1]],
        y1=n[ax],   y2=n[cross[0]],   y3=n[cross[1]],
        x2=x[cross[0]], x3=x[cross[1]],
    )
    return N, d[0], d[1], dA, c


def _c33_floor(y3, floor=1e-3):
    """C33 = n.b3 appears in the drilling denominators; floor it away from zero
    exactly like the validated 1-D code floors 1/xdot2 (vertical walls)."""
    return y3 if abs(y3) > floor else (floor if y3 >= 0 else -floor)


def _omega3_ops(N, D1, D2, c, k22):
    """Drilling operators (A.3/A.5): row vectors over the element DOFs.

        omega_3      = OM3h . w  + OM3l . w'  + OM3e . eb          (A.3)
        omega'_3     =             OM3pl . w' (same row as OM3h)   (A.5)
                       + the -(C3b/C33) omega'_b algebraic part
    Returns (OM3h(20,), OM3l(20,), OM3e(4,), OM3p_l(20,)).
    """
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.array([x11, c["x21"], c["x31"]])            # x_{i;1}
    xi2 = np.array([x12, c["x22"], c["x32"]])            # x_{i;2}
    yv = np.array([c["y1"], c["y2"], c["y3"]])           # C_{3i}
    C33 = _c33_floor(c["y3"])
    Rn1 = c["x2"] * c["x31"] - c["x3"] * c["x21"]        # (A.4)
    Rn2 = c["x2"] * c["x32"] - c["x3"] * c["x22"]

    OM3h = np.zeros(4 * NDOF); OM3l = np.zeros(4 * NDOF)
    OM3pl = np.zeros(4 * NDOF); OM3e = np.zeros(4)
    #   S = k1 (x11 Rn2 - x12 Rn1) + w'_i (x11 x_{i;2} - x12 x_{i;1})
    #       + (w_{i|1} x_{i;2} - w_{i|2} x_{i;1})              -> /(2 C33)
    OM3e[1] = (x11 * Rn2 - x12 * Rn1) / (2 * C33)
    for a in range(4):
        o = NDOF * a
        for i in range(3):
            OM3h[o + i] += (xi2[i] * D1[a] - xi1[i] * D2[a]) / (2 * C33)
            OM3l[o + i] += (x11 * xi2[i] - x12 * xi1[i]) * N[a] / (2 * C33)
            #   omega'_3 (A.5): (w'_{i|1} x_{i;2} - w'_{i|2} x_{i;1})/(2C33)
            OM3pl[o + i] += (xi2[i] * D1[a] - xi1[i] * D2[a]) / (2 * C33)
        #   -(C3b/C33) omega_b   (algebraic; beta = GLOBAL 1,2 -> dofs om1,om2)
        OM3h[o + 3] += -(yv[0] / C33) * N[a]
        OM3h[o + 4] += -(yv[1] / C33) * N[a]
        OM3pl[o + 3] += -(yv[0] / C33) * N[a]             # -(C3b/C33) omega'_b
        OM3pl[o + 4] += -(yv[1] / C33) * N[a]
    return OM3_SIGN*OM3h, OM3_SIGN*OM3l, OM3_SIGN*OM3e, OM3_SIGN*OM3pl


def _lambda_ops(N, D1, D2, c, k22, alpha):
    """Lambda_alpha = omega'_3 x_{1;alpha} + omega_{3|alpha}   (A.6)

    omega_{3|alpha} = d/dzeta_alpha of the omega_3 operator:
      (i)  shape part:  -(C3b/C33) omega_{b|alpha}   [KEPT: first derivative]
           w_{i|1|alpha}, w_{i|2|alpha}              [SECOND derivative: DROPPED
                                                      -- C0-RM cannot resolve]
      (ii) coefficient part (alpha=2 only; hoop Frenet with wall curvature k22):
             (x_{i;2})_{;2} = k22 y_i     (y_i)_{;2} = -k22 x_{i;2}
             (x_i)_{;2}     = x_{i;2}     (x_{i;1})_{;2} = 0  [geodesic: dropped]
           all (.)_{;1} coefficient derivatives (frame taper rate): DROPPED
           (higher order; flat-ish quads).
    Returns (Lh(20,), Ll(20,), Le(4,)) acting on (w, w', eb).
    """
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.array([x11, c["x21"], c["x31"]])
    xi2 = np.array([x12, c["x22"], c["x32"]])
    yv = np.array([c["y1"], c["y2"], c["y3"]])
    C33 = _c33_floor(c["y3"])
    x1a = x11 if alpha == 1 else x12                      # x_{1;alpha}
    Da = D1 if alpha == 1 else D2

    OM3h, OM3l, OM3e, OM3pl = _omega3_ops(N, D1, D2, c, k22)
    Lh = np.zeros(4 * NDOF); Ll = np.zeros(4 * NDOF); Le = np.zeros(4)
    sgn = OM3_SIGN                                        # every Lambda term carries omega_3's sign

    # ---- omega'_3 x_{1;alpha} : acts on the w' field ----
    Ll += x1a * OM3pl                                     # (OM3pl already sign-scaled)

    # ---- omega_{3|alpha}, shape part: -(C3b/C33) omega_{b|alpha} ----
    for a in range(4):
        o = NDOF * a
        Lh[o + 3] += -sgn * (yv[0] / C33) * Da[a]
        Lh[o + 4] += -sgn * (yv[1] / C33) * Da[a]

    if alpha == 2:
        # ---- omega_{3|2}, coefficient part (hoop Frenet, k22) ----
        Rn2 = c["x2"] * c["x32"] - c["x3"] * c["x22"]
        Rn1 = c["x2"] * c["x31"] - c["x3"] * c["x21"]
        # d/dz2 of 1/(2C33):  (1/2C33)_{;2} = +k22 x32 / (2 C33^2)
        dinvC = k22 * c["x32"] / (2 * C33 * C33)
        # d/dz2 of Rn2 = (x2)_{;2} x32 + x2 k22 y3 - (x3)_{;2} x22 - x3 k22 y2
        #             = (x22 x32 - x32 x22 = 0) + k22 (x2 y3 - x3 y2)
        dRn2 = k22 * (c["x2"] * c["y3"] - c["x3"] * c["y2"])
        # d/dz2 of Rn1 (x_{i;1} frozen): x22 x31 - x32 x21
        dRn1 = c["x22"] * c["x31"] - c["x32"] * c["x21"]
        # S-coefficient derivatives -> macro (k1) row
        Le[1] += sgn * ((x11 * dRn2 - x12 * dRn1) / (2 * C33)
                        + (x11 * Rn2 - x12 * Rn1) * dinvC)
        for a in range(4):
            o = NDOF * a
            for i in range(3):
                # w'-coefficient: d/dz2 [ (x11 xi2 - x12 xi1)/(2C33) ]
                #   (xi2)_{;2} = k22 y_i ; xi1 frozen
                Ll[o + i] += sgn * ((x11 * k22 * yv[i]) * N[a] / (2 * C33)
                                    + (x11 * xi2[i] - x12 * xi1[i]) * N[a] * dinvC)
                # w_{i|1},w_{i|2}-coefficient derivatives (shape part stays 1st-order)
                Lh[o + i] += sgn * ((k22 * yv[i]) * D1[a] / (2 * C33)
                                    + (xi2[i] * D1[a] - xi1[i] * D2[a]) * dinvC)
            # -(C3b/C33) coefficient derivative:
            #   (yb/C33)_{;2} = (-k22 x_{b;2} C33 - yb (-k22 x32)) / C33^2
            for b, (yb, xb2) in enumerate(((yv[0], c["x12"]), (yv[1], c["x22"]))):
                dcoef = (-k22 * xb2 * C33 + yb * k22 * c["x32"]) / (C33 * C33)
                Lh[o + 3 + b] += -sgn * dcoef * N[a]
    return Lh, Ll, Le


def quad_ops_general(X, e1m, e2m, e3m, xi, eta, k22, cross=(1, 2), ax=None):
    """GENERAL RM operators at (xi,eta):  returns
        BDe (6,4)  macro map        Gamma_eps (D-block rows)
        BDh (6,20) fluctuation      Gamma_h
        BDl (6,20) x1-derivative    Gamma_l
        BGe (2,4), BGh (2,20), BGl (2,20)   transverse-shear blocks
        dA  surface Jacobian
    Rows: [e11, e22, 2e12, k11, k22, k12+21] and [2g13, 2g23]."""
    if ax is None:
        ax = [j for j in range(3) if j not in cross][0]
    N, D1, D2, dA, c = _surf_frame(X, e3m, xi, eta, cross, ax)
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.array([x11, c["x21"], c["x31"]])
    xi2 = np.array([x12, c["x22"], c["x32"]])
    yv = np.array([c["y1"], c["y2"], c["y3"]])
    x2, x3 = c["x2"], c["x3"]
    Rn1 = x2 * c["x31"] - x3 * c["x21"]
    Rn2 = x2 * c["x32"] - x3 * c["x22"]

    BDe = np.zeros((6, 4)); BDh = np.zeros((6, 4 * NDOF)); BDl = np.zeros((6, 4 * NDOF))
    BGe = np.zeros((2, 4)); BGh = np.zeros((2, 4 * NDOF)); BGl = np.zeros((2, 4 * NDOF))

    # ================= MEMBRANE (general; paper Shell Strains) =================
    # e11 = x11^2 g11 + x11(x2 x31 - x3 x21)k1 + x11^2(x3 k2 - x2 k3)
    BDe[0] = [x11**2, x11 * Rn1, x11**2 * x3, -(x11**2) * x2]
    # e22 = x12^2 g11 + x12 Rn2 k1 + x12^2 (x3 k2 - x2 k3)
    BDe[1] = [x12**2, x12 * Rn2, x12**2 * x3, -(x12**2) * x2]
    # 2e12 = 2 x11 x12 g11 + (x11 Rn2 + x12 Rn1) k1 + 2 x11 x12 (x3 k2 - x2 k3)
    BDe[2] = [2 * x11 * x12, x11 * Rn2 + x12 * Rn1, 2 * x11 * x12 * x3, -2 * x11 * x12 * x2]
    for a in range(4):
        o = NDOF * a
        for i in range(3):
            BDh[0, o + i] += xi1[i] * D1[a]                       # x_{i;1} w_{i|1}
            BDl[0, o + i] += x11 * xi1[i] * N[a]                  # x11 x_{i;1} w'_i
            BDh[1, o + i] += xi2[i] * D2[a]                       # x_{i;2} w_{i|2}
            BDl[1, o + i] += x12 * xi2[i] * N[a]                  # x12 x_{i;2} w'_i
            BDh[2, o + i] += xi1[i] * D2[a] + xi2[i] * D1[a]      # mixed
            BDl[2, o + i] += (x11 * xi2[i] + x12 * xi1[i]) * N[a]

    # ================= CURVATURES (A.8, Lambda per A.6) ========================
    L1h, L1l, L1e = _lambda_ops(N, D1, D2, c, k22, alpha=1)
    L2h, L2l, L2e = _lambda_ops(N, D1, D2, c, k22, alpha=2)

    # k11 = x11 x_{i;2} k_{1i} + x_{b;2} x11 om'_b + x_{b;2} om_{b|1} + x32 Lambda_1
    BDe[3] = [0.0, x11 * x12, x11 * c["x22"], x11 * c["x32"]]
    BDe[3] += c["x32"] * L1e
    for a in range(4):
        o = NDOF * a
        BDl[3, o + 3] += x11 * c["x12"] * N[a]                    # x_{1;2} x11 om'_1
        BDl[3, o + 4] += x11 * c["x22"] * N[a]                    # x_{2;2} x11 om'_2
        BDh[3, o + 3] += c["x12"] * D1[a]                         # x_{1;2} om_{1|1}
        BDh[3, o + 4] += c["x22"] * D1[a]                         # x_{2;2} om_{2|1}
    BDh[3] += c["x32"] * L1h; BDl[3] += c["x32"] * L1l

    # k22 = -[ x12 x_{i;1} k_{1i} + x_{b;1} x12 om'_b + x_{b;1} om_{b|2} + x31 Lambda_2 ]
    BDe[4] = [0.0, -x12 * x11, -x12 * c["x21"], -x12 * c["x31"]]
    BDe[4] += -c["x31"] * L2e
    for a in range(4):
        o = NDOF * a
        BDl[4, o + 3] += -x12 * c["x11"] * N[a]
        BDl[4, o + 4] += -x12 * c["x21"] * N[a]
        BDh[4, o + 3] += -c["x11"] * D2[a]
        BDh[4, o + 4] += -c["x21"] * D2[a]
    BDh[4] += -c["x31"] * L2h; BDl[4] += -c["x31"] * L2l
    # NOTE prismatic check: x12=0, x31=0 -> k22 = -x_{1;1} om_{1|2} = -omdot_1  (eq:prism)
    # [the VALIDATED 1-D code carries +omdot_1 with its own compensating sign
    #  convention for omega_1; the prismatic-identity test decides the sign map]

    # k12+21 = k_{1i}(x12 x_{i;2} - x11 x_{i;1}) + om'_b(x12 x_{b;2} - x11 x_{b;1})
    #          + (x_{b;2} om_{b|2} - x_{b;1} om_{b|1}) + x32 Lambda_2 - x31 Lambda_1
    BDe[5] = [0.0,
              x12 * x12 - x11 * x11,
              x12 * c["x22"] - x11 * c["x21"],
              x12 * c["x32"] - x11 * c["x31"]]
    BDe[5] += c["x32"] * L2e - c["x31"] * L1e
    for a in range(4):
        o = NDOF * a
        BDl[5, o + 3] += (x12 * c["x12"] - x11 * c["x11"]) * N[a]
        BDl[5, o + 4] += (x12 * c["x22"] - x11 * c["x21"]) * N[a]
        BDh[5, o + 3] += c["x12"] * D2[a] - c["x11"] * D1[a]
        BDh[5, o + 4] += c["x22"] * D2[a] - c["x21"] * D1[a]
    BDh[5] += c["x32"] * L2h - c["x31"] * L1h
    BDl[5] += c["x32"] * L2l - c["x31"] * L1l

    # ================= TRANSVERSE SHEAR ========================================
    # minimal general form (prismatic-consistent):
    #   2g13 = y_i w_{i|1} + omega_2 + x11 y_i w'_i     [TODO: full eq12/drilling-
    #   2g23 = y_i w_{i|2} - omega_1                     boosted general gamma_13]
    for a in range(4):
        o = NDOF * a
        for i in range(3):
            BGh[0, o + i] += yv[i] * D1[a]
            BGl[0, o + i] += x11 * yv[i] * N[a]
            BGh[1, o + i] += yv[i] * D2[a]
        BGh[0, o + 4] += N[a]
        BGh[1, o + 3] += -N[a]
    return BDe, BDh, BDl, BGe, BGh, BGl, dA


# ------------------------------------------------------------------ MITC tying
# Dvorkin-Bathe MITC4 assumed transverse shear, same tying points as the
# validated reduced element: g23 sampled at (0,-1),(0,+1) linear in eta;
# g13 sampled at (-1,0),(+1,0) linear in xi; BOTH rows tied (mitc_both).
_TIE_G23 = [(0.0, -1.0), (0.0, 1.0)]
_TIE_G13 = [(-1.0, 0.0), (1.0, 0.0)]


def _mitc_shear_general(X, e1m, e2m, e3m, xi, eta, k22, cross, ax):
    r23 = [quad_ops_general(X, e1m, e2m, e3m, tx, te, k22, cross, ax)[4][1:2, :]
           for (tx, te) in _TIE_G23]
    r13 = [quad_ops_general(X, e1m, e2m, e3m, tx, te, k22, cross, ax)[4][0:1, :]
           for (tx, te) in _TIE_G13]
    g23 = 0.5 * (1.0 - eta) * r23[0] + 0.5 * (1.0 + eta) * r23[1]
    g13 = 0.5 * (1.0 - xi) * r13[0] + 0.5 * (1.0 + xi) * r13[1]
    return np.vstack([g13, g23])


# -------------------------------------------------------------------- assembly
def assemble_segment_general(nodes, quads, subdom, e1s, e2s, e3s, D_by, G_by,
                             k22_e, cross=(1, 2)):
    """Assemble the six MSG blocks with the GENERAL RM taper operators:
        Dhh = <Gh' C Gh>   Dhe = <Gh' C Ge>   Dee = <Ge' C Ge>
        Dhl = <Gh' C Gl>   Dll = <Gl' C Gl>   Dle = <Gl' C Ge>
    each block carrying BOTH the classical (6x6 ABD, D) and the transverse-shear
    (2x2, G) energies; the shear h-rows are MITC-tied."""
    ax = [j for j in range(3) if j not in cross][0]
    Nn = len(nodes); ndof = NDOF * Nn
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 4)); Dee = np.zeros((4, 4))
    Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof)); Dle = np.zeros((ndof, 4))
    gp = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)),
          (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]
    for q, quad in enumerate(quads):
        X = nodes[quad]; k22 = float(k22_e[q])
        D = D_by[int(subdom[q])] if not isinstance(D_by, dict) or int(subdom[q]) in D_by else D_by[subdom[q]]
        G = G_by[int(subdom[q])]
        g = np.concatenate([[NDOF * n + c for c in range(NDOF)] for n in quad])
        for (xi, eta) in gp:
            BDe, BDh, BDl, BGe, BGh, BGl, dA = quad_ops_general(
                X, e1s[q], e2s[q], e3s[q], xi, eta, k22, cross, ax)
            BGt = _mitc_shear_general(X, e1s[q], e2s[q], e3s[q], xi, eta, k22, cross, ax)
            w = dA  # unit gauss weights on [-1,1]^2 (2x2)
            Dhh[np.ix_(g, g)] += (BDh.T @ D @ BDh + BGt.T @ G @ BGt) * w
            Dhe[g] += (BDh.T @ D @ BDe + BGt.T @ G @ BGe) * w
            Dee += (BDe.T @ D @ BDe + BGe.T @ G @ BGe) * w
            Dhl[np.ix_(g, g)] += (BDh.T @ D @ BDl + BGt.T @ G @ BGl) * w
            Dll[np.ix_(g, g)] += (BDl.T @ D @ BDl + BGl.T @ G @ BGl) * w
            Dle[g] += (BDl.T @ D @ BDe + BGl.T @ G @ BGe) * w
    return Dhh, Dhe, Dee, Dhl, Dll, Dle
