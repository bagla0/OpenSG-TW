"""segment_indep.py -- EXPERIMENTAL: omega_3 as an INDEPENDENT 6th DOF.

The general operator (segment_element_general) ELIMINATES the drilling omega_3 via
  omega_3 = S/(2 C33) - (C3b/C33) omega_b ,   C33 = n.b3 ,
which is singular on flat walls where C33=n.b3 == 0 (the GA3-carrier walls / shear web).
Here omega_3 is kept as a genuine nodal DOF (per node [w1,w2,w3,om1,om2,om3]) and appears
DIRECTLY -- no 1/C33 anywhere:

  curvature:  Lambda_a = omega_3|a + x_{1;a} omega_3'          (was the eliminated Lambda)
  shear:      2g13 += C23 om3 = x_{3;2} om3 ,  2g23 += -C13 om3 = -x_{3;1} om3
  membrane:   unchanged (no omega_3)

The in-plane symmetry that DEFINED omega_3 is re-imposed in its FINITE (undivided) form
as a penalty on the drilling residual
  DR = C33 om3 + C3b om_b - S/2   (= C33 (om3 - om3_eliminated), finite even at C33=0),
  S/2 = 1/2[ k1(x11 Rn2 - x12 Rn1) + w'_i(x11 x_{i;2}-x12 x_{i;1}) + (w_{i|1}x_{i;2}-w_{i|2}x_{i;1}) ].
On healthy walls (C33~1) a large penalty pins om3 to its eliminated value (recovering the
general result); where C33=0 the residual drops om3 (constrains om_b instead) and om3 is set
by its own curvature stiffness -- no singularity.  PEN sweeps the penalty weight.

Boundary: reuse the validated 5-DOF rings (ring_general) for dofs [0..4]; om3 (dof 5) is left
FREE at the boundary (natural BC).
"""
import numpy as np
from segment_element import _bilinear, dirichlet_solve
from segment_element_general import _surf_frame

NDOF6 = 6


def quad_ops_indep(X, e3m, xi, eta, k22, cross, ax, kg=0.0):
    """8-strain 6-DOF operators + drilling-residual DR at (xi,eta).
    Returns BDe(6,4),BDh(6,24),BDl(6,24), BGe(2,4),BGh(2,24),BGl(2,24),
            DRe(4),DRh(24),DRl(24), dA."""
    N, D1, D2, dA, c = _surf_frame(X, e3m, xi, eta, cross, ax)
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.array([x11, c["x21"], c["x31"]])            # x_{i;1}
    xi2 = np.array([x12, c["x22"], c["x32"]])            # x_{i;2}
    yv = np.array([c["y1"], c["y2"], c["y3"]])           # C_{3i}
    x2, x3 = c["x2"], c["x3"]
    x21, x31, x22, x32 = c["x21"], c["x31"], c["x22"], c["x32"]
    y1, y2, y3 = c["y1"], c["y2"], c["y3"]
    Rn1 = x2 * x31 - x3 * x21
    Rn2 = x2 * x32 - x3 * x22
    n = NDOF6

    BDe = np.zeros((6, 4)); BDh = np.zeros((6, 4 * n)); BDl = np.zeros((6, 4 * n))
    BGe = np.zeros((2, 4)); BGh = np.zeros((2, 4 * n)); BGl = np.zeros((2, 4 * n))
    DRe = np.zeros(4); DRh = np.zeros(4 * n); DRl = np.zeros(4 * n)

    # ---------------- MEMBRANE (unchanged; no omega_3) ----------------
    BDe[0] = [x11**2, x11 * Rn1, x11**2 * x3, -(x11**2) * x2]
    BDe[1] = [x12**2, x12 * Rn2, x12**2 * x3, -(x12**2) * x2]
    BDe[2] = [2 * x11 * x12, x11 * Rn2 + x12 * Rn1, 2 * x11 * x12 * x3, -2 * x11 * x12 * x2]
    for a in range(4):
        o = n * a
        for i in range(3):
            BDh[0, o + i] += xi1[i] * D1[a]
            BDl[0, o + i] += x11 * xi1[i] * N[a]
            BDh[1, o + i] += xi2[i] * D2[a]
            BDl[1, o + i] += x12 * xi2[i] * N[a]
            BDh[2, o + i] += xi1[i] * D2[a] + xi2[i] * D1[a]
            BDl[2, o + i] += (x11 * xi2[i] + x12 * xi1[i]) * N[a]

    # ---------------- CURVATURE (non-Lambda parts as general; Lambda = DIRECT om3) --------
    BDe[3] = [0.0, x11 * x12, x11 * c["x22"], x11 * c["x32"]]
    BDe[4] = [0.0, -x12 * x11, -x12 * x21, -x12 * x31]
    BDe[5] = [0.0, x12 * x12 - x11 * x11, x12 * x22 - x11 * x21, x12 * x32 - x11 * x31]
    for a in range(4):
        o = n * a
        # k11 : x_{b;2} x11 om'_b + x_{b;2} om_{b|1}
        BDl[3, o + 3] += x11 * x12 * N[a]; BDl[3, o + 4] += x11 * x22 * N[a]
        BDh[3, o + 3] += x12 * D1[a];      BDh[3, o + 4] += x22 * D1[a]
        # k22 : -(x_{b;1} x12 om'_b + x_{b;1} om_{b|2})
        BDl[4, o + 3] += -x12 * x11 * N[a]; BDl[4, o + 4] += -x12 * x21 * N[a]
        BDh[4, o + 3] += -x11 * D2[a];      BDh[4, o + 4] += -x21 * D2[a]
        # k12 : om'_b(x12 x_{b;2}-x11 x_{b;1}) + (x_{b;2}om_{b|2}-x_{b;1}om_{b|1})
        BDl[5, o + 3] += (x12 * x12 - x11 * x11) * N[a]; BDl[5, o + 4] += (x12 * x22 - x11 * x21) * N[a]
        BDh[5, o + 3] += x12 * D2[a] - x11 * D1[a];      BDh[5, o + 4] += x22 * D2[a] - x21 * D1[a]
        # Lambda contributions via DIRECT om3 (dof 5):  L_a = om3|a + x_{1;a} om3'
        #   k11 += x32 L1 ; k22 += -x31 L2 ; k12 += x32 L2 - x31 L1
        BDh[3, o + 5] += x32 * D1[a];               BDl[3, o + 5] += x32 * x11 * N[a]
        BDh[4, o + 5] += -x31 * D2[a];              BDl[4, o + 5] += -x31 * x12 * N[a]
        BDh[5, o + 5] += x32 * D2[a] - x31 * D1[a]; BDl[5, o + 5] += (x32 * x12 - x31 * x11) * N[a]

    # ---------------- TRANSVERSE SHEAR (pre-elimination: NO 1/C33; DIRECT om3) ------------
    swept = x2 * y3 - x3 * y2                            # k1 coeff without 1/C33 (h33=0)
    BGe[0] = np.array([x11 * y1, x11 * swept, x11 * y1 * x3, -x11 * y1 * x2])
    BGe[1] = np.array([x12 * y1, x12 * swept, x12 * y1 * x3, -x12 * y1 * x2])
    for a in range(4):
        o = n * a
        for i in range(3):
            BGh[0, o + i] += yv[i] * D1[a]               # C3i w_{i|1}
            BGh[1, o + i] += yv[i] * D2[a]               # C3i w_{i|2}
            BGl[0, o + i] += x11 * yv[i] * N[a]          # chain rule
            BGl[1, o + i] += x12 * yv[i] * N[a]
        BGh[0, o + 3] += x12 * N[a]; BGh[0, o + 4] += x22 * N[a]    # C_2a om_a
        BGh[1, o + 3] += -x11 * N[a]; BGh[1, o + 4] += -x21 * N[a]  # -C_1a om_a
        BGh[0, o + 5] += x32 * N[a]                     # +C23 om3
        BGh[1, o + 5] += -x31 * N[a]                    # -C13 om3

    # ---------------- DRILLING RESIDUAL DR = C33 om3 + C3b om_b - S/2 (finite) -------------
    DRe[1] = -0.5 * (x11 * Rn2 - x12 * Rn1)
    for a in range(4):
        o = n * a
        for i in range(3):
            DRh[o + i] += -0.5 * (xi2[i] * D1[a] - xi1[i] * D2[a])          # -1/2 (w_{i|1}x_{i;2}-w_{i|2}x_{i;1})
            DRl[o + i] += -0.5 * (x11 * xi2[i] - x12 * xi1[i]) * N[a]       # -1/2 w'_i(x11 x_{i;2}-x12 x_{i;1})
        DRh[o + 3] += y1 * N[a]; DRh[o + 4] += y2 * N[a]; DRh[o + 5] += y3 * N[a]  # C3b om_b + C33 om3
    return BDe, BDh, BDl, BGe, BGh, BGl, DRe, DRh, DRl, dA


# standard Dvorkin-Bathe MITC4 tying points (same as segment_element_general):
#   gamma_13 (row 0): sampled at (-1,0),(+1,0), linear in xi
#   gamma_23 (row 1): sampled at (0,-1),(0,+1), linear in eta
_TIE_G13 = [(-1.0, 0.0), (1.0, 0.0)]
_TIE_G23 = [(0.0, -1.0), (0.0, 1.0)]


def _mitc_shear_indep(X, e3m, xi, eta, k22, cross, ax, kg=0.0, scheme="mitc4_g23"):
    """Tied (assumed-strain) transverse-shear BGh rows (2 x 4*NDOF6) at (xi,eta).

    With the INDEPENDENT drilling omega_3 the gamma_13 row is ALGEBRAIC in omega_3 on
    flat walls (x_{3;2} omega_3 -- the role omega_2 plays prismatically), so tying it
    removes the director penalization (hourglass; GA3 -31/-50% on the thin square).
    The field-consistent selective scheme ties ONLY the locking-prone gamma_23 row
    ('mitc4_g23', mirroring the validated prismatic element); 'mitc4_both' kept for
    the ablation."""
    ops = quad_ops_indep(X, e3m, xi, eta, k22, cross, ax, kg)
    r23 = [quad_ops_indep(X, e3m, tx, te, k22, cross, ax, kg)[4][1:2, :] for (tx, te) in _TIE_G23]
    g23 = 0.5 * (1.0 - eta) * r23[0] + 0.5 * (1.0 + eta) * r23[1]
    if scheme == "mitc4_both":
        r13 = [quad_ops_indep(X, e3m, tx, te, k22, cross, ax, kg)[4][0:1, :] for (tx, te) in _TIE_G13]
        g13 = 0.5 * (1.0 - xi) * r13[0] + 0.5 * (1.0 + xi) * r13[1]
    else:
        g13 = ops[4][0:1, :]
    return np.vstack([g13, g23])


def _d_scale(D_by):
    """Characteristic ABD stiffness magnitude = max |diag(D)| over layups.  The drilling
    penalty is set to beta*this so it is dimensionally commensurate with the elastic
    stiffness (enforces DR=0 without the ill-conditioning of an over-large absolute pen)."""
    keys = D_by.keys() if isinstance(D_by, dict) else range(len(D_by))
    return max(float(np.max(np.abs(np.diag(np.asarray(D_by[k]))))) for k in keys)


def assemble_segment_indep(nodes, quads, subdom, e3s, D_by, G_by, k22_e, cross, ax,
                           kg_e=None, pen=None, pen_beta=0.1, dof_map=None,
                           shear="full"):
    """6-DOF assembly with the finite drilling-residual penalty pen*DR^2.
    pen defaults to pen_beta * max|diag(D)| (D-scaled; robust across materials/thickness).
    shear: 'full' (default) integrates both transverse-shear rows untied -- with the
    INDEPENDENT omega_3 both rows carry algebraic drilling content that Dvorkin-Bathe
    assumed-strain interpolation ALIASES (square thin: 'mitc4_both' -31/-50% GA3,
    'mitc4_g23' -15/-30% GA2, vs 'full' -4..-6%), and no transverse-shear locking is
    observed untied (circle: full==tied to 0.2%).  Tied variants kept for the ablation."""
    if pen is None:
        pen = pen_beta * _d_scale(D_by)
    if dof_map is None:
        dof_map = np.arange(len(nodes))
    Nn = int(np.max(dof_map)) + 1; ndof = NDOF6 * Nn
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 4)); Dee = np.zeros((4, 4))
    Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof)); Dle = np.zeros((ndof, 4))
    gpv = 1.0 / np.sqrt(3.0)
    gp = [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]
    for q, quad in enumerate(quads):
        X = nodes[quad]; k22 = float(k22_e[q]); kg = float(kg_e[q]) if kg_e is not None else 0.0
        D = D_by[int(subdom[q])]; G = G_by[int(subdom[q])]
        g = np.concatenate([[NDOF6 * int(dof_map[nd]) + cc for cc in range(NDOF6)] for nd in quad])
        gij = (g[:, None], g[None, :])
        for (xi, eta) in gp:
            BDe, BDh, BDl, BGe, BGh, BGl, DRe, DRh, DRl, dA = quad_ops_indep(
                X, e3s[q], xi, eta, k22, cross, ax, kg)
            BGt = BGh if shear == "full" else _mitc_shear_indep(
                X, e3s[q], xi, eta, k22, cross, ax, kg, scheme=shear)
            w = dA
            DRh2 = DRh[:, None]; DRl2 = DRl[:, None]
            np.add.at(Dhh, gij, (BDh.T @ D @ BDh + BGt.T @ G @ BGt + pen * (DRh2 @ DRh2.T)) * w)
            np.add.at(Dhe, g, (BDh.T @ D @ BDe + BGt.T @ G @ BGe + pen * (DRh2 @ DRe[None, :])).squeeze() * w)
            Dee += (BDe.T @ D @ BDe + BGe.T @ G @ BGe + pen * np.outer(DRe, DRe)) * w
            np.add.at(Dhl, gij, (BDh.T @ D @ BDl + BGt.T @ G @ BGl + pen * (DRh2 @ DRl2.T)) * w)
            np.add.at(Dll, gij, (BDl.T @ D @ BDl + BGl.T @ G @ BGl + pen * (DRl2 @ DRl2.T)) * w)
            np.add.at(Dle, g, (BDl.T @ D @ BDe + BGl.T @ G @ BGe + pen * (DRl2 @ DRe[None, :])).squeeze() * w)
    return Dhh, Dhe, Dee, Dhl, Dll, Dle


def assemble_constraint(nodes, quads, subdom, e3s, k22_e, cross, ax, kg_e=None,
                        lam_space="elem", dof_map=None):
    """Weak drilling-constraint operators for the Lagrange multiplier field.

    lam_space='elem' (default): one PIECEWISE-CONSTANT multiplier per element,
      <DR>_e = 0 -- the inf-sup-stable choice (an equal-order nodal multiplier
      over-constrains under refinement: square thin GA2/GA3 drift -1/-2% ->
      -17/-9% from NC=24 to NC=96, the classical LBB failure of equal-order
      multiplier spaces; the element-constant space removes the drift).
    lam_space='node': one multiplier per node, <N_a DR> = 0 (kept for the ablation).
    lam_space='elem_nofold': element-constant multipliers, EXCLUDING elements adjacent
      to a FOLD line (nodes where incident-element normals disagree > 30 deg).  The
      drilling constraint is derived on a smooth surface patch; across a slope
      discontinuity the C0-shared fields cannot satisfy both walls' symmetry rows
      simultaneously, and the growing number of fold-line rows (~1/h) produces the
      linear-in-1/h over-constraint drift seen on the square (GA2 -1 -> -17%).
    dof_map (optional): node -> dof-node index (wrapped prismatic strip, as in the
    ring SG).  With a dof_map the local index vector contains REPEATED dofs, so the
    accumulation uses np.add.at (fancy-index += silently drops duplicates).
    Returns G (P x 6Nd) on w_s, Gl (P x 6Nd) on w_s', Ge (P x 4) on eb."""
    Nn = len(nodes)
    if dof_map is None:
        dof_map = np.arange(Nn)
    Nd = int(np.max(dof_map)) + 1; M = NDOF6 * Nd
    skip = np.zeros(len(quads), bool)
    if lam_space == "elem_nofold":
        # per-element unit normal from the two diagonals
        nrm = np.cross(nodes[quads[:, 2]] - nodes[quads[:, 0]],
                       nodes[quads[:, 3]] - nodes[quads[:, 1]])
        nrm /= (np.linalg.norm(nrm, axis=1)[:, None] + 1e-30)
        node_ref = [[] for _ in range(Nn)]
        for q, quad in enumerate(quads):
            for nd in quad:
                node_ref[int(nd)].append(q)
        fold_node = np.zeros(Nn, bool)
        for nd in range(Nn):
            qs = node_ref[nd]
            for a in range(len(qs)):
                for c in range(a + 1, len(qs)):
                    if abs(float(nrm[qs[a]] @ nrm[qs[c]])) < np.cos(np.radians(30.0)):
                        fold_node[nd] = True
        skip = np.array([any(fold_node[int(nd)] for nd in quad) for quad in quads])
    if lam_space.startswith("elem"):
        row_of = -np.ones(len(quads), int)
        row_of[~skip] = np.arange(int((~skip).sum()))
        P = int((~skip).sum())
    else:
        P = Nd
    G = np.zeros((P, M)); Gl = np.zeros((P, M)); Ge = np.zeros((P, 4))
    gpv = 1.0 / np.sqrt(3.0)
    gp = [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]
    for q, quad in enumerate(quads):
        if skip[q]:
            continue
        X = nodes[quad]; k22 = float(k22_e[q]); kg = float(kg_e[q]) if kg_e is not None else 0.0
        gloc = np.array([NDOF6 * int(dof_map[nd]) + c for nd in quad for c in range(NDOF6)])
        for (xi, eta) in gp:
            N, _, _ = _bilinear(xi, eta)
            _, _, _, _, _, _, DRe, DRh, DRl, dA = quad_ops_indep(X, e3s[q], xi, eta, k22, cross, ax, kg)
            if lam_space.startswith("elem"):
                rr = row_of[q] if lam_space == "elem_nofold" else q
                np.add.at(G[rr], gloc, DRh * dA)
                np.add.at(Gl[rr], gloc, DRl * dA)
                Ge[rr] += DRe * dA
            else:
                for a in range(4):
                    nd = int(dof_map[quad[a]])
                    np.add.at(G[nd], gloc, N[a] * DRh * dA)
                    np.add.at(Gl[nd], gloc, N[a] * DRl * dA)
                    Ge[nd] += N[a] * DRe * dA
    return G, Gl, Ge


def build_C_Psi_segment6(nodes, quads, cross):
    """6-DOF rigid-body kernel/constraints (om3 column = 0: drilling is not a rigid mode)."""
    Nn = len(nodes); ndof = NDOF6 * Nn
    C = np.zeros((4, ndof)); Psi = np.zeros((ndof, 4))
    gpv = 1.0 / np.sqrt(3.0)
    qp = [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]
    for quad in quads:
        X = nodes[quad]
        for (xi, eta) in qp:
            Nn_, dNx, dNe = _bilinear(xi, eta)
            dA = np.linalg.norm(np.cross(dNx @ X, dNe @ X))
            for a, nd in enumerate(quad):
                for cc in range(4):
                    C[cc, NDOF6 * nd + cc] += Nn_[a] * dA
    for nd in range(Nn):
        y2, y3 = nodes[nd, cross[0]], nodes[nd, cross[1]]
        Psi[NDOF6 * nd + 0, 0] = 1.0
        Psi[NDOF6 * nd + 1, 1] = 1.0
        Psi[NDOF6 * nd + 2, 2] = 1.0
        Psi[NDOF6 * nd + 1, 3] = -y3; Psi[NDOF6 * nd + 2, 3] = y2; Psi[NDOF6 * nd + 3, 3] = -1.0
    return C, Psi
