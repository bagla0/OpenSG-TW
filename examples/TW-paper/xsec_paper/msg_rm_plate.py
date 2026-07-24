"""msg_rm_plate.py -- MSG-based Reissner-Mindlin plate law + recovery (Yu-2002 construction).

SEPARATE analysis module (does not modify the existing pipeline).  Implements, on the same
1-D through-thickness SG discretization as ``opensg_jax.fe_jax.msg_materials.compute_ABD_matrix``:

  zeroth order   : V0 warping  -> A6 (classical ABD; must reproduce compute_ABD_matrix exactly)
  first order    : gradient-driven warping columns C1bar, C2bar (Yu 2002 Eq. 38: v = Cbar_a E,_a)
  second order   : gradient energy H (12x12 blocks over [E,1; E,2], Yu Eq. 41)
  RM projection  : least-squares minimization of the residual U* over the shear compliance
                   X = G^{-1} (3) plus the relaxed-constraint constants c_a (36), Yu Eq. 49-55
                   -> the MSG transverse-shear stiffness  G_msg (2x2)
  recovery       : through-thickness 3-D strain incl. the first-order (equilibrium-consistent)
                   transverse-shear terms, given the plate strains E and their in-plane
                   gradients E,1 (axial) and E,2 (arc).

Voigt strain order [11,22,33,23,13,12]; plate strain order E = [e11,e22,g12,k11,k22,k12];
transverse shear gamma = [2g13, 2g23].  x = through-thickness coordinate measured from the
bottom (OML) face minus z_ref, exactly as compute_ABD_matrix.

Validation (run this file):  homogeneous isotropic -> G_msg = 5/6 G h;  orthotropic laminates
-> G_msg ~= Whitney/complementary-energy transverse_shear_stiffness; A6 == compute_ABD_matrix.
"""
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from opensg_jax.fe_jax.msg_materials import (rotated_stiffness_6x6, _lagrange_dN, _plate_B,
                                             compute_ABD_matrix)
from opensg_jax.fe_jax.msg_transverse_shear import transverse_shear_stiffness


def _lagrange_N(nodes_xi, xi):
    npn = len(nodes_xi)
    N = np.ones(npn)
    for i in range(npn):
        for j in range(npn):
            if j != i:
                N[i] *= (xi - nodes_xi[j]) / (nodes_xi[i] - nodes_xi[j])
    return N


def _grad_ops(nodes_xi, xi):
    """M1, M2 (6 x 3*(p+1)): strain contribution of the IN-PLANE gradient of the warping.
    w,1: e11 += w1,1 ; 2g13 += w3,1 ; g12 += w2,1
    w,2: e22 += w2,2 ; 2g23 += w3,2 ; g12 += w1,2
    """
    N = _lagrange_N(nodes_xi, xi)
    npn = len(nodes_xi)
    M1 = np.zeros((6, 3 * npn)); M2 = np.zeros((6, 3 * npn))
    for n in range(npn):
        M1[0, 3 * n + 0] = N[n]      # eps11 <- w1,1
        M1[4, 3 * n + 2] = N[n]      # 2g13  <- w3,1
        M1[5, 3 * n + 1] = N[n]      # g12   <- w2,1
        M2[1, 3 * n + 1] = N[n]      # eps22 <- w2,2
        M2[3, 3 * n + 2] = N[n]      # 2g23  <- w3,2
        M2[5, 3 * n + 0] = N[n]      # g12   <- w1,2
    return M1, M2


def rm_plate_msg(thick, angles_deg, mat_names, material_db, n_per_layer=4, elem_order=3,
                 z_ref=0.0):
    """Build the MSG-RM plate law.  Returns a dict:
    A6 (6x6 ABD), G_msg (2x2), H (12x12), V0/C1bar/C2bar (ndofs x 6), node_x, elem_layer,
    C_layers, elem_order, Ustar_rel (residual after projection / before), X (=G^{-1})."""
    nlay = len(thick)
    layer_bot = np.concatenate([[0.0], np.cumsum(thick)])
    C_layers = [rotated_stiffness_6x6(material_db[mat_names[k]]['E'],
                                      material_db[mat_names[k]]['G'],
                                      material_db[mat_names[k]]['nu'],
                                      angles_deg[k]) for k in range(nlay)]
    p = int(elem_order)
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    n_elem = nlay * n_per_layer
    n_node = p * n_elem + 1
    ndofs = 3 * n_node

    node_x = np.empty(n_node)
    elem_layer = np.empty(n_elem, dtype=int)
    idx = 0
    for k in range(nlay):
        for s in range(n_per_layer):
            xl = layer_bot[k] + thick[k] * s / n_per_layer
            xr = layer_bot[k] + thick[k] * (s + 1) / n_per_layer
            for j in range(p):
                node_x[p * idx + j] = xl + (xr - xl) * j / p
            elem_layer[idx] = k
            idx += 1
    node_x[p * n_elem] = layer_bot[-1]
    node_x = node_x - z_ref

    xi_g, w_g = np.polynomial.legendre.leggauss(max(3, p + 1))

    K = np.zeros((ndofs, ndofs))
    F = np.zeros((ndofs, 6))
    D_ee = np.zeros((6, 6))
    T1 = np.zeros((ndofs, ndofs)); T2 = np.zeros((ndofs, ndofs))     # int B^T C M_a
    U1 = np.zeros((ndofs, ndofs)); U2 = np.zeros((ndofs, ndofs))     # int M_a^T C B
    W11 = np.zeros((ndofs, ndofs)); W12 = np.zeros((ndofs, ndofs)); W22 = np.zeros((ndofs, ndofs))
    P1e = np.zeros((ndofs, 6)); P2e = np.zeros((ndofs, 6))           # int M_a^T C Ge

    for e in range(n_elem):
        xl = node_x[p * e]; xr = node_x[p * e + p]; he = xr - xl
        Ck = C_layers[elem_layer[e]]
        dofs = np.arange(3 * p * e, 3 * p * e + 3 * (p + 1))
        for q in range(len(xi_g)):
            xi = xi_g[q]
            x_q = 0.5 * (xl + xr) + 0.5 * he * xi
            dw = 0.5 * he * w_g[q]
            B = _plate_B(nodes_xi, xi, he)
            M1, M2 = _grad_ops(nodes_xi, xi)
            Ge = np.zeros((6, 6))
            Ge[0, 0] = 1.0; Ge[0, 3] = x_q
            Ge[1, 1] = 1.0; Ge[1, 4] = x_q
            Ge[5, 2] = 1.0; Ge[5, 5] = x_q
            ix = np.ix_(dofs, dofs)
            K[ix] += B.T @ Ck @ B * dw
            F[dofs, :] += B.T @ Ck @ Ge * dw
            D_ee += Ge.T @ Ck @ Ge * dw
            T1[ix] += B.T @ Ck @ M1 * dw; T2[ix] += B.T @ Ck @ M2 * dw
            U1[ix] += M1.T @ Ck @ B * dw; U2[ix] += M2.T @ Ck @ B * dw
            W11[ix] += M1.T @ Ck @ M1 * dw
            W12[ix] += M1.T @ Ck @ M2 * dw
            W22[ix] += M2.T @ Ck @ M2 * dw
            P1e[dofs, :] += M1.T @ Ck @ Ge * dw
            P2e[dofs, :] += M2.T @ Ck @ Ge * dw

    null = np.zeros((ndofs, 3))
    null[0::3, 0] = 1.0; null[1::3, 1] = 1.0; null[2::3, 2] = 1.0
    Q, _ = np.linalg.qr(null)
    Pp = np.eye(ndofs) - Q @ Q.T
    beta = np.max(np.abs(np.diag(K)))
    K_proj = Pp @ K @ Pp + beta * (Q @ Q.T)

    V0 = np.linalg.solve(K_proj, -(Pp @ F)); V0 = Pp @ V0
    A6 = D_ee + V0.T @ F

    # first-order driver R_a = (int B^T C M_a) V0 - (int M_a^T C B) V0 - int M_a^T C Ge
    #   term (i)  : cross of B v with the gradient strain M_a(V0) E,_a
    #   term (ii) : the parts-shifted S0-cross  -M_a^T C (B V0 + Ge)
    R1 = T1 @ V0 - (U1 @ V0 + P1e)
    R2 = T2 @ V0 - (U2 @ V0 + P2e)

    C1bar = np.linalg.solve(K_proj, -(Pp @ R1)); C1bar = Pp @ C1bar
    C2bar = np.linalg.solve(K_proj, -(Pp @ R2)); C2bar = Pp @ C2bar

    # pure gradient quadratic Q_ab = V0^T (int M_a^T C M_b) V0  (+ Ge cross with M_a V0)
    Q11 = V0.T @ W11 @ V0
    Q12 = V0.T @ W12 @ V0
    Q22 = V0.T @ W22 @ V0
    H11 = Q11 + R1.T @ C1bar
    H12 = Q12 + 0.5 * (R1.T @ C2bar + C1bar.T @ R2)
    H22 = Q22 + R2.T @ C2bar
    H11 = 0.5 * (H11 + H11.T); H22 = 0.5 * (H22 + H22.T)
    H = np.block([[H11, H12], [H12.T, H22]])

    # ---- RM projection (Yu 2002 sec. 4): E = R - D1 g,1 - D2 g,2 ; equilibrium swap;
    #      LS over X = G^{-1} (sym 2x2) and relaxed constants c1,c2 (3x6 each) ----
    D1 = np.zeros((6, 2)); D2 = np.zeros((6, 2))
    D1[3, 0] = 1.0; D1[5, 1] = 1.0        # k11 <- 2g13,1 ; k12 <- 2g23,1
    D2[4, 1] = 1.0; D2[5, 0] = 1.0        # k22 <- 2g23,2 ; k12 <- 2g13,2
    S1 = null.T @ R1                       # (3,6) effect of constant shifts
    S2 = null.T @ R2

    AD1 = A6 @ D1; AD2 = A6 @ D2          # (6,2)
    scl = float(np.max(np.abs(H)) + 1e-30)

    def blocks(X, c1, c2):
        Bs = H11 + AD1 @ X @ AD1.T + c1.T @ S1 + S1.T @ c1
        Cs = H12 + AD1 @ X @ AD2.T + c1.T @ S2 + S1.T @ c2
        Ds = H22 + AD2 @ X @ AD2.T + c2.T @ S2 + S2.T @ c2
        return np.block([[Bs, Cs], [Cs.T, Ds]])

    # linear LS: unknown p = [x11,x12,x22, c1(18), c2(18)] -> minimize ||blocks||_F
    nun = 3 + 36
    Amat = np.zeros((144, nun)); b0 = -blocks(np.zeros((2, 2)), np.zeros((3, 6)), np.zeros((3, 6))).ravel()
    for j in range(nun):
        pj = np.zeros(nun); pj[j] = 1.0
        X = np.array([[pj[0], pj[1]], [pj[1], pj[2]]])
        c1 = pj[3:21].reshape(3, 6); c2 = pj[21:39].reshape(3, 6)
        Amat[:, j] = blocks(X, c1, c2).ravel() + b0               # linear response of column j
    # column scaling for conditioning
    cs = np.linalg.norm(Amat, axis=0); cs[cs == 0] = 1.0
    sol = np.linalg.lstsq(Amat / cs, b0, rcond=None)[0] / cs
    X = np.array([[sol[0], sol[1]], [sol[1], sol[2]]])
    c1 = sol[3:21].reshape(3, 6); c2 = sol[21:39].reshape(3, 6)
    res = blocks(X, c1, c2)
    Ustar_rel = float(np.linalg.norm(res) / (np.linalg.norm(H) + 1e-30))

    ev = np.linalg.eigvalsh(X)
    if ev.min() <= 0:
        G_msg = None
    else:
        G_msg = np.linalg.inv(X)

    return {"A6": A6, "G_msg": G_msg, "X": X, "H": H, "Ustar_rel": Ustar_rel,
            "V0": V0, "C1bar": C1bar, "C2bar": C2bar, "node_x": node_x,
            "elem_layer": elem_layer, "C_layers": C_layers, "elem_order": p,
            "angles": list(angles_deg), "c1": c1, "c2": c2}


def msgrm_strain_at_depth(obj, z, E6, dE1=None, dE2=None):
    """3-D Voigt strain at through-thickness x=z (same origin as node_x) including the
    first-order gradient terms.  E6 = plate strains, dE1/dE2 = in-plane gradients of E6
    (axial / arc).  Returns (Gam6, Sig6, ply_angle_deg)."""
    dE1 = np.zeros(6) if dE1 is None else np.asarray(dE1, float)
    dE2 = np.zeros(6) if dE2 is None else np.asarray(dE2, float)
    node_x = obj["node_x"]; p = obj["elem_order"]
    n_elem = len(obj["elem_layer"])
    e = int(np.clip(np.searchsorted(node_x[::p][1:], z, side="right"), 0, n_elem - 1))
    xl = node_x[p * e]; xr = node_x[p * e + p]; he = xr - xl
    xi = np.clip(2.0 * (z - xl) / he - 1.0, -1.0, 1.0)
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    B = _plate_B(nodes_xi, xi, he)
    M1, M2 = _grad_ops(nodes_xi, xi)
    Ge = np.zeros((6, 6))
    x_q = 0.5 * (xl + xr) + 0.5 * he * xi
    Ge[0, 0] = 1.0; Ge[0, 3] = x_q
    Ge[1, 1] = 1.0; Ge[1, 4] = x_q
    Ge[5, 2] = 1.0; Ge[5, 5] = x_q
    dofs = np.arange(3 * p * e, 3 * p * e + 3 * (p + 1))
    w_loc = (obj["V0"][dofs] @ E6 + obj["C1bar"][dofs] @ dE1 + obj["C2bar"][dofs] @ dE2)
    g1 = obj["V0"][dofs] @ dE1; g2 = obj["V0"][dofs] @ dE2
    Gam = B @ w_loc + Ge @ E6 + M1 @ g1 + M2 @ g2
    k = obj["elem_layer"][e]
    Sig = obj["C_layers"][k] @ Gam
    return Gam, Sig, obj["angles"][k]


if __name__ == "__main__":
    # ---- validation 1: homogeneous isotropic -> G = 5/6 G h ----
    mdb = {"iso": {"E": [70e9] * 3, "G": [70e9 / 2.6] * 3, "nu": [0.3] * 3, "rho": 1.0}}
    h = 0.01
    r = rm_plate_msg([h], [0.0], ["iso"], mdb, n_per_layer=4, z_ref=h / 2)
    Gh = 70e9 / 2.6 * h
    print("iso: G_msg/(Gh) diag =", None if r["G_msg"] is None else np.diag(r["G_msg"]) / Gh,
          " target 5/6 = %.6f   Ustar_rel %.3e" % (5.0 / 6.0, r["Ustar_rel"]))
    Gw = transverse_shear_stiffness([h], [0.0], ["iso"], mdb)[0]
    print("     Whitney diag/(Gh) =", np.diag(Gw) / Gh)
    mdb0 = {"iso0": {"E": [70e9] * 3, "G": [35e9] * 3, "nu": [0.0] * 3, "rho": 1.0}}
    r0 = rm_plate_msg([h], [0.0], ["iso0"], mdb0, n_per_layer=4, z_ref=h / 2)
    print("     nu=0 : G_msg/(Gh) =", None if r0["G_msg"] is None else np.diag(r0["G_msg"]) / (35e9 * h),
          " Ustar %.2e" % r0["Ustar_rel"])
    rf = rm_plate_msg([h], [0.0], ["iso"], mdb, n_per_layer=12, elem_order=3, z_ref=h / 2)
    print("     fine : G_msg/(Gh) =", None if rf["G_msg"] is None else np.diag(rf["G_msg"]) / Gh)
    A_ref = compute_ABD_matrix([h], [0.0], ["iso"], mdb, n_per_layer=4)[0]
    print("     |A6 - compute_ABD| =", np.max(np.abs(r["A6"] - A_ref)))

    # ---- validation 2: [0/90/0] Pagano-style graphite/epoxy ----
    mdb2 = {"gr": {"E": [172.4e9, 6.89e9, 6.89e9], "G": [3.45e9, 1.38e9, 3.45e9],
                   "nu": [0.25, 0.25, 0.25], "rho": 1.0}}
    thk = [0.005, 0.005, 0.005]; ang = [0.0, 90.0, 0.0]; mats = ["gr"] * 3
    r2 = rm_plate_msg(thk, ang, mats, mdb2, n_per_layer=4, z_ref=sum(thk) / 2)
    Gw2 = transverse_shear_stiffness(thk, ang, mats, mdb2)[0]
    print("[0/90/0]: G_msg =", None if r2["G_msg"] is None else np.array2string(r2["G_msg"], precision=4))
    print("          Whitney=", np.array2string(Gw2, precision=4), "  Ustar_rel %.3e" % r2["Ustar_rel"])

    # ---- validation 3: web sandwich biax/foam/biax (s10 materials) ----
    mdb3 = {"biax": {"E": [11.5e9, 11.5e9, 1.3e10], "G": [11.8e9, 3.5e9, 3.5e9],
                     "nu": [0.5, 0.09, 0.09], "rho": 1.0},
            "foam": {"E": [1.42e8] * 3, "G": [6.0e7] * 3, "nu": [0.2] * 3, "rho": 1.0}}
    thk = [0.002, 0.042, 0.002]; ang = [0.0] * 3; mats = ["biax", "foam", "biax"]
    r3 = rm_plate_msg(thk, ang, mats, mdb3, n_per_layer=4, z_ref=sum(thk) / 2)
    Gw3 = transverse_shear_stiffness(thk, ang, mats, mdb3)[0]
    print("web sandwich: G_msg =", None if r3["G_msg"] is None else np.array2string(r3["G_msg"], precision=4))
    print("              Whitney=", np.array2string(Gw3, precision=4), "  Ustar_rel %.3e" % r3["Ustar_rel"])
