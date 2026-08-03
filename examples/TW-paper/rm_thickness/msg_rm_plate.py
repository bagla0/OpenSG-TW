"""msg_rm_plate.py -- MSG Reissner-Mindlin plate law + through-thickness recovery.

Standalone copy of ``examples/TW-paper/xsec_paper/msg_rm_plate.py`` with the OpenSG
imports replaced by the local ``materials`` module, plus TWO additions needed for the
through-thickness study:

  * ``msgrm_recover_profile`` -- vectorised through-thickness sampling that honours ply
    interfaces (one-sided limits on both sides of every interface);
  * ``sigma33_equilibrium``   -- transverse normal stress by through-thickness integration
    of the 3-D equilibrium equation sigma33,3 = -sigma13,1 - sigma23,2, using the
    first-order (equilibrium-consistent) transverse shear recovered by MSG.  FSDT cannot
    produce this quantity at all; in Garg et al. (2023) it is supplied by a GPR surrogate
    trained on 3-D elasticity solutions.

Construction (Yu, Hodges & Volovoi 2002 IJSS 39:5185; Yu 2005 IJSS 42:6680):
  zeroth order : V0 warping                       -> A6 (classical ABD)
  first order  : gradient warping C1bar, C2bar    (v = Cbar_a E,_a)
  second order : gradient energy H (12x12)
  RM projection: least squares of the residual U* over X = G^{-1} and the relaxed
                 constants -> the MSG transverse-shear stiffness G_msg (2x2)

Voigt strain order [11,22,33,23,13,12]; plate strain E = [e11,e22,g12,k11,k22,k12];
transverse shear gamma = [2g13, 2g23].  x is measured from the BOTTOM face minus z_ref.
"""
import numpy as np

from materials import rotated_stiffness_6x6, plate_B as _plate_B, grad_ops as _grad_ops


def rm_plate_msg(thick, angles_deg, mat_names, material_db, n_per_layer=4, elem_order=3,
                 z_ref=0.0):
    """Build the MSG-RM plate law.  See module docstring for the returned keys."""
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
    T1 = np.zeros((ndofs, ndofs)); T2 = np.zeros((ndofs, ndofs))
    U1 = np.zeros((ndofs, ndofs)); U2 = np.zeros((ndofs, ndofs))
    W11 = np.zeros((ndofs, ndofs)); W12 = np.zeros((ndofs, ndofs)); W22 = np.zeros((ndofs, ndofs))
    P1e = np.zeros((ndofs, 6)); P2e = np.zeros((ndofs, 6))

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

    R1 = T1 @ V0 - (U1 @ V0 + P1e)
    R2 = T2 @ V0 - (U2 @ V0 + P2e)

    C1bar = np.linalg.solve(K_proj, -(Pp @ R1)); C1bar = Pp @ C1bar
    C2bar = np.linalg.solve(K_proj, -(Pp @ R2)); C2bar = Pp @ C2bar

    Q11 = V0.T @ W11 @ V0
    Q12 = V0.T @ W12 @ V0
    Q22 = V0.T @ W22 @ V0
    H11 = Q11 + R1.T @ C1bar
    H12 = Q12 + 0.5 * (R1.T @ C2bar + C1bar.T @ R2)
    H22 = Q22 + R2.T @ C2bar
    H11 = 0.5 * (H11 + H11.T); H22 = 0.5 * (H22 + H22.T)
    H = np.block([[H11, H12], [H12.T, H22]])

    D1 = np.zeros((6, 2)); D2 = np.zeros((6, 2))
    D1[3, 0] = 1.0; D1[5, 1] = 1.0
    D2[4, 1] = 1.0; D2[5, 0] = 1.0
    S1 = null.T @ R1
    S2 = null.T @ R2

    AD1 = A6 @ D1; AD2 = A6 @ D2

    def blocks(X, c1, c2):
        Bs = H11 + AD1 @ X @ AD1.T + c1.T @ S1 + S1.T @ c1
        Cs = H12 + AD1 @ X @ AD2.T + c1.T @ S2 + S1.T @ c2
        Ds = H22 + AD2 @ X @ AD2.T + c2.T @ S2 + S2.T @ c2
        return np.block([[Bs, Cs], [Cs.T, Ds]])

    nun = 3 + 36
    Amat = np.zeros((144, nun))
    b0 = -blocks(np.zeros((2, 2)), np.zeros((3, 6)), np.zeros((3, 6))).ravel()
    for j in range(nun):
        pj = np.zeros(nun); pj[j] = 1.0
        X = np.array([[pj[0], pj[1]], [pj[1], pj[2]]])
        c1 = pj[3:21].reshape(3, 6); c2 = pj[21:39].reshape(3, 6)
        Amat[:, j] = blocks(X, c1, c2).ravel() + b0
    cs = np.linalg.norm(Amat, axis=0); cs[cs == 0] = 1.0
    sol = np.linalg.lstsq(Amat / cs, b0, rcond=None)[0] / cs
    X = np.array([[sol[0], sol[1]], [sol[1], sol[2]]])
    c1 = sol[3:21].reshape(3, 6); c2 = sol[21:39].reshape(3, 6)
    res = blocks(X, c1, c2)
    Ustar_rel = float(np.linalg.norm(res) / (np.linalg.norm(H) + 1e-30))

    ev = np.linalg.eigvalsh(X)
    G_msg = None if ev.min() <= 0 else np.linalg.inv(X)

    return {"A6": A6, "G_msg": G_msg, "X": X, "H": H, "Ustar_rel": Ustar_rel,
            "V0": V0, "C1bar": C1bar, "C2bar": C2bar, "node_x": node_x,
            "elem_layer": elem_layer, "C_layers": C_layers, "elem_order": p,
            "angles": list(angles_deg), "c1": c1, "c2": c2,
            "thick": np.asarray(thick, float), "z_ref": float(z_ref)}


def _elem_of(obj, z):
    node_x = obj["node_x"]; p = obj["elem_order"]
    n_elem = len(obj["elem_layer"])
    return int(np.clip(np.searchsorted(node_x[::p][1:], z, side="right"), 0, n_elem - 1))


def msgrm_strain_at_depth(obj, z, E6, dE1=None, dE2=None, elem=None):
    """3-D Voigt strain/stress at through-thickness x = z including the first-order
    gradient terms.  E6 = plate strains, dE1/dE2 = their in-plane gradients."""
    dE1 = np.zeros(6) if dE1 is None else np.asarray(dE1, float)
    dE2 = np.zeros(6) if dE2 is None else np.asarray(dE2, float)
    node_x = obj["node_x"]; p = obj["elem_order"]
    e = _elem_of(obj, z) if elem is None else int(elem)
    xl = node_x[p * e]; xr = node_x[p * e + p]; he = xr - xl
    xi = np.clip(2.0 * (z - xl) / he - 1.0, -1.0, 1.0)
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    B = _plate_B(nodes_xi, xi, he)
    M1, M2 = _grad_ops(nodes_xi, xi)
    x_q = 0.5 * (xl + xr) + 0.5 * he * xi
    Ge = np.zeros((6, 6))
    Ge[0, 0] = 1.0; Ge[0, 3] = x_q
    Ge[1, 1] = 1.0; Ge[1, 4] = x_q
    Ge[5, 2] = 1.0; Ge[5, 5] = x_q
    dofs = np.arange(3 * p * e, 3 * p * e + 3 * (p + 1))
    w_loc = (obj["V0"][dofs] @ E6 + obj["C1bar"][dofs] @ dE1 + obj["C2bar"][dofs] @ dE2)
    g1 = obj["V0"][dofs] @ dE1
    g2 = obj["V0"][dofs] @ dE2
    Gam = B @ w_loc + Ge @ E6 + M1 @ g1 + M2 @ g2
    k = obj["elem_layer"][e]
    Sig = obj["C_layers"][k] @ Gam
    return Gam, Sig, obj["angles"][k]


def msgrm_recover_profile(obj, E6, dE1=None, dE2=None, n_per_layer=41, eps=1e-9):
    """Through-thickness profile honouring ply interfaces.

    Returns (z, Gam, Sig) with z on the SAME origin as ``obj['node_x']`` (i.e. already
    referred to z_ref) and one-sided limits present on both sides of every interface.
    """
    node_x = obj["node_x"]; p = obj["elem_order"]
    thick = obj["thick"]
    bot = np.concatenate([[0.0], np.cumsum(thick)]) - obj["z_ref"]
    zs = []
    for k in range(len(thick)):
        a, b = bot[k], bot[k + 1]
        zs.append(np.linspace(a + eps * (b - a), b - eps * (b - a), n_per_layer))
    zs = np.concatenate(zs)
    Gam = np.empty((zs.size, 6))
    Sig = np.empty((zs.size, 6))
    for i, z in enumerate(zs):
        # pick the element on the correct side of an interface
        e = _elem_of(obj, z)
        Gam[i], Sig[i], _ = msgrm_strain_at_depth(obj, z, E6, dE1, dE2, elem=e)
    return zs, Gam, Sig


def sigma33_equilibrium(z, s13, s23=None, p1=1.0, p2=0.0):
    """sigma33 by integrating  sigma33,3 = -sigma13,1 - sigma23,2  from the bottom face.

    For a field with x1-dependence cos(p1 x) for sigma13 and sin(p1 x) for sigma33 (the
    cylindrical-bending Navier form), sigma13,1 = -p1 * s13 * sin, so the amplitude obeys
    d(sigma33)/dz = p1 * s13.  ``z`` must be monotonically increasing and include both
    one-sided limits at interfaces (which contribute zero width).
    """
    integrand = p1 * np.asarray(s13, float)
    if s23 is not None:
        integrand = integrand + p2 * np.asarray(s23, float)
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    out[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z))
    return out
