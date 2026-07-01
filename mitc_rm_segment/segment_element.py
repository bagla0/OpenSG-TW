"""
segment_element.py     [ Windows opensg_2_0_env ]
========================================================================
PART 2: the 2-D MITC4 Reissner-Mindlin SHELL element for the 3D-SG segment.

Design principle (verified against the FEniCS shell operators gamma_h/gamma_l
and the validated 1-D RM msg_rm_timo):
  * DOF/node = [w1,w2,w3, omega1,omega2]  (5), bilinear quad -> 20 DOF/elem.
  * Gamma_h (BDq) = the FULL surface gradient of the warping in the local frame
    (axial d/dx1 = grad.e1, hoop d/ds = grad.e2), with RM rotation-based bending.
    When the field is span-invariant (d/dx1 -> 0) BDq reduces EXACTLY to the 1-D
    BDq -- that is the prismatic self-check.
  * transverse shear (BGq rows [2eps13, 2eps23]) uses MITC4 tying
    (Dvorkin-Bathe / Chapelle-Bathe sec 8.2):
        gamma23 (xi-shear)  tied at (xi=0, eta=+/-1), linear in eta;
        gamma13 (eta-shear) tied at (xi=+/-1, eta=0), linear in xi.
  * Gamma_l (BLq) = the 1-D cross-section shear-warping operator (== FEniCS
    gamma_l for this frame).
  * macro map Gamma_e = _macro_BD (reused from msg_rm).

The segment is solved with the boundary rings' warping imposed as DIRICHLET BCs
(not periodicity): rigid-body modes are fixed by the boundary, exactly like the
FEniCS compute_stiffness.  For a PRISMATIC cylinder the interior warping must be
span-invariant, i.e. equal to the ring warping at every axial station -- the
acceptance gate you asked for.
"""

import numpy as np
from opensg_jax.fe_jax.msg_rm import _macro_BD


# --------------------------------------------------------------- quad kinematics
def _bilinear(xi, eta):
    """4-node bilinear N and parametric derivatives at (xi,eta) in [-1,1]^2.
    Corner order matches the mesh winding [ (j,k),(j,k+1),(j+1,k+1),(j+1,k) ] ->
    reference corners [(-1,-1),(1,-1),(1,1),(-1,1)] (xi ~ hoop, eta ~ axial)."""
    xc = np.array([-1., 1., 1., -1.]); ec = np.array([-1., -1., 1., 1.])
    N = 0.25 * (1 + xc * xi) * (1 + ec * eta)
    dNx = 0.25 * xc * (1 + ec * eta)
    dNe = 0.25 * ec * (1 + xc * xi)
    return N, dNx, dNe


def _quad_ops(X, e1, e2, e3, xi, eta, k22):
    """Return BDq(6,20), BGq(2,20), BLq(6,20) and geometry at (xi,eta).

    X (4,3) node coords; e1/e2/e3 unit frame.  In-frame derivatives d/dx1 (axial,
    along e1) and d/ds (hoop, along e2) are obtained from the 2x2 metric
    G = [[e1.Jxi, e1.Jeta],[e2.Jxi, e2.Jeta]] :  [d/dx1; d/ds] = G^-1 [dN/dxi; dN/deta].
    dA = |Jxi x Jeta|.
    """
    N, dNx, dNe = _bilinear(xi, eta)
    Jxi = dNx @ X                       # dX/dxi (3,)
    Jeta = dNe @ X                      # dX/deta (3,)
    # chain rule: [df/dxi; df/deta] = G^T [df/dx1; df/ds]  with
    # G = [[e1.Jxi, e1.Jeta],[e2.Jxi, e2.Jeta]]  ->  [df/dx1; df/ds] = (G^T)^-1 [df/dxi; df/deta].
    # (Using G^-1 instead of (G^T)^-1 swaps the hoop/axial lengths for a non-square element.)
    G = np.array([[e1 @ Jxi, e1 @ Jeta],
                  [e2 @ Jxi, e2 @ Jeta]])
    # per-node in-frame derivatives: rows = nodes, cols = [d/dx1, d/ds]
    d = (np.linalg.inv(G.T) @ np.vstack([dNx, dNe])).T   # (4,2): d[a,0]=dNa/dx1, d[a,1]=dNa/ds
    dA = np.linalg.norm(np.cross(Jxi, Jeta))
    x = N @ X
    x2, x3 = x[1], x[2]                            # cross-section coords (y,z)
    t2, t3 = e2[1], e2[2]                          # hoop tangent (y,z) -- as in 1-D
    n2, n3 = t3, -t2                               # 1-D in-plane normal convention

    BDq = np.zeros((6, 20)); BGq = np.zeros((2, 20)); BLq = np.zeros((6, 20))
    for a in range(4):
        o = 5 * a
        D1, Ds, Na = d[a, 0], d[a, 1], N[a]       # d/dx1, d/ds, shape value
        # --- Gamma_h : membrane (full surface gradient) ---
        BDq[0, o+0] += D1                                   # eps11 = dw1/dx1        (axial; ->0 span-inv)
        BDq[1, o+1] += t2*Ds; BDq[1, o+2] += t3*Ds          # eps22 = d(t.w)/ds
        BDq[2, o+0] += Ds                                   # 2eps12 = dw1/ds
        BDq[2, o+1] += t2*D1; BDq[2, o+2] += t3*D1          # 2eps12 += d(t.w)/dx1   (axial; ->0)
        # --- Gamma_h : bending (RM, first derivative of rotations) ---
        BDq[3, o+4] += D1                                   # k11 = domega2/dx1      (axial; ->0)
        BDq[4, o+3] += Ds                                   # k22 = domega1/ds
        BDq[5, o+4] += Ds; BDq[5, o+0] += 0.5*k22*Ds        # 2k12 = domega2/ds + 0.5 k22 dw1/ds
        BDq[5, o+3] += D1                                   # 2k12 += domega1/dx1    (axial; ->0)
        # --- transverse shear (pre-tying; MITC4 assembles the tied form) ---
        BGq[0, o+4] += Na                                   # 2eps13 = omega2
        BGq[0, o+1] += n2*D1; BGq[0, o+2] += n3*D1          # 2eps13 += d(n.w)/dx1   (axial; ->0)
        BGq[1, o+1] += n2*Ds; BGq[1, o+2] += n3*Ds; BGq[1, o+3] += -Na  # 2eps23 = d(n.w)/ds - omega1
        # --- Gamma_l : shear-warping surrogate (== 1-D BLq / FEniCS gamma_l) ---
        BLq[0, o+0] += Na
        BLq[2, o+1] += t2*Na; BLq[2, o+2] += t3*Na
        BLq[5, o+1] += 2*t3*Ds - 0.5*k22*t2*Na
        BLq[5, o+2] += -2*t2*Ds - 0.5*k22*t3*Na
    return BDq, BGq, BLq, (x2, x3, t2, t3, dA)


# tying points (Dvorkin-Bathe MITC4): row1 (gamma23, xi-shear) at (0,-1),(0,+1);
#                                      row0 (gamma13, eta-shear) at (-1,0),(+1,0)
_TIE = {"g23": [(0.0, -1.0), (0.0, 1.0)], "g13": [(-1.0, 0.0), (1.0, 0.0)]}


def _mitc_shear(X, e1, e2, e3, xi, eta, k22):
    """Assumed (tied) transverse-shear 2x20 operator BGb at (xi,eta)."""
    (A23), (B23) = [_quad_ops(X, e1, e2, e3, tx, te, k22)[1][1:2, :] for (tx, te) in _TIE["g23"]]
    (A13), (B13) = [_quad_ops(X, e1, e2, e3, tx, te, k22)[1][0:1, :] for (tx, te) in _TIE["g13"]]
    g23 = 0.5*(1.0 - eta)*A23 + 0.5*(1.0 + eta)*B23      # linear in eta
    g13 = 0.5*(1.0 - xi)*A13 + 0.5*(1.0 + xi)*B13        # linear in xi
    return np.vstack([g13, g23])


# ------------------------------------------------------------------- assembly
def assemble_segment(nodes, quads, subdom, e1s, e2s, e3s, D_by, G_by, k22_by):
    """Assemble Dhh, Dhe, Dee, Dhl, Dll, Dle for the 2-D quad segment.

    nodes (Nn,3), quads (Ne,4) 0-based, subdom (Ne,), e{1,2,3}s (Ne,3),
    D_by/G_by/k22_by keyed by subdomain id.  2x2 Gauss on the D/Gamma_l energy;
    the transverse-shear G-energy uses MITC4-tied BGb at the same points.
    """
    Nn = len(nodes); ndof = 5 * Nn
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 4)); Dee = np.zeros((4, 4))
    Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof)); Dle = np.zeros((ndof, 4))
    gp = 1.0 / np.sqrt(3.0)
    quad_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    for e, quad in enumerate(quads):
        X = nodes[quad]                                     # (4,3)
        e1, e2, e3 = e1s[e], e2s[e], e3s[e]
        D = D_by[int(subdom[e])]; G = G_by[int(subdom[e])]; k22 = float(k22_by[int(subdom[e])])
        g = np.concatenate([[5*n, 5*n+1, 5*n+2, 5*n+3, 5*n+4] for n in quad])
        for (xi, eta) in quad_pts:
            BDq, BGq, BLq, geo = _quad_ops(X, e1, e2, e3, xi, eta, k22)
            x2, x3, t2, t3, dA = geo
            BGb = _mitc_shear(X, e1, e2, e3, xi, eta, k22)
            BDe = _macro_BD(x2, x3, t2, t3, k22)
            BGe = np.zeros((2, 4))
            Dhh[np.ix_(g, g)] += (BDq.T @ D @ BDq + BGb.T @ G @ BGb) * dA
            Dhe[g] += (BDq.T @ D @ BDe + BGb.T @ G @ BGe) * dA
            Dee += (BDe.T @ D @ BDe + BGe.T @ G @ BGe) * dA
            Dhl[np.ix_(g, g)] += BDq.T @ D @ BLq * dA
            Dll[np.ix_(g, g)] += BLq.T @ D @ BLq * dA
            Dle[g] += BLq.T @ D @ BDe * dA
    return Dhh, Dhe, Dee, Dhl, Dll, Dle


# --------------------------------------- rigid kernel + constraints (segment EB)
def build_C_Psi_segment(nodes, quads):
    """4 rigid-body modes (3 translations + twist) and the conjugate <.>=0
    constraints, integrated over the 2-D segment area -- the 2-D analogue of
    msg_rm_timo.build_C_Psi, so the element-agnostic msg_solver KKT solve applies
    unchanged.  Psi twist uses the node's cross-section coords (y,z)."""
    Nn = len(nodes); ndof = 5 * Nn
    C = np.zeros((4, ndof)); Psi = np.zeros((ndof, 4))
    gp = 1.0 / np.sqrt(3.0)
    qp = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    for quad in quads:
        X = nodes[quad]
        for (xi, eta) in qp:
            N, dNx, dNe = _bilinear(xi, eta)
            dA = np.linalg.norm(np.cross(dNx @ X, dNe @ X))
            for a, nd in enumerate(quad):
                for c in range(4):                    # <w1>=<w2>=<w3>=<omega1>=0
                    C[c, 5 * nd + c] += N[a] * dA
    for nd in range(Nn):
        y2, y3 = nodes[nd, 1], nodes[nd, 2]
        Psi[5*nd+0, 0] = 1.0                          # w1 translation
        Psi[5*nd+1, 1] = 1.0                          # w2 translation
        Psi[5*nd+2, 2] = 1.0                          # w3 translation
        Psi[5*nd+1, 3] = -y3; Psi[5*nd+2, 3] = y2; Psi[5*nd+3, 3] = -1.0   # twist
    return C, Psi


# ---------------------------------------------- Dirichlet (boundary) segment solve
def dirichlet_solve(K, RHS, bdofs, bvals):
    """Solve K u = RHS with u[bdofs] = bvals (columns = load cases).

    Partitioned: K_ii u_i = RHS_i - K_ib bvals ; u[bdofs]=bvals.  Returns u (ndof,nc).
    """
    ndof = K.shape[0]; nc = RHS.shape[1]
    free = np.setdiff1d(np.arange(ndof), bdofs)
    u = np.zeros((ndof, nc)); u[bdofs] = bvals
    Kii = K[np.ix_(free, free)]
    rhs = RHS[free] - K[np.ix_(free, bdofs)] @ bvals
    u[free] = np.linalg.solve(Kii, rhs)
    return u
