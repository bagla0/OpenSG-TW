"""analytical_timo_ring.py -- ANALYTICAL Timoshenko constants of the center-ref circular
RM tube by the Fourier-harmonic reduction of the VAM chain (zeroth + first order + the
generalized-Timoshenko condensation), mirroring the paper's operators exactly.

Continuous ring on the circle (radius R, prismatic, center reference):
  fields u(th) = (w1,w2,w3,om1,om2,om3), multiplier lam(th); harmonics n=0..NH.
  strain rows (paper eq:membrane/curvature/shear, global components, on the circle):
    e11 = g11 + x3 k2 - x2 k3 + w1'
    e22 = t.d2 w            (t = (-sin, cos), d2 = (1/R) d/dth)
    g12 = R k1 + d2 w1 + t.w'
    k11 = t2 k2 + t3 k3 + t.om'(2,3)
    k22 = -d2 om1
    k12 = -k1 - om1' + t.d2 om(2,3)
    g13 = y.w'(2,3) + t.om(2,3)          (swept = 0 centric)
    g23 = y.d2 w(2,3) - om1              (y = (-cos, -sin), inward)
  drilling residual g = y.om(2,3) - (R/2) k1 + (1/2) d2 w1 - (1/2) t.w'(2,3) = 0.

PHASE A (numeric): assemble the harmonic KKT, run the VAM chain (eq:zeroth, eq:first,
eq:ABCD, eq:transform), compare the 6 diagonal Timoshenko constants against ring_indep
on the same wall law (iso and [-45] ply, R/h = 2 and 10) -- the validation.
PHASE B (symbolic, iso): same chain in sympy -> closed forms for EA, GA, GJ, EI.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for q in (MITC, REPO, HERE):
    sys.path.insert(0, q)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

NH = 3          # harmonics 0..NH per field
NF = 6          # w1,w2,w3,om1,om2,om3
NLAM = True     # include multiplier field


def basis_size(nh):
    return 1 + 2 * nh                      # a0, (ak, bk) k=1..nh


def eval_basis(th, nh):
    """values and theta-derivatives of [1, cos k th, sin k th]."""
    V = [np.ones_like(th)]
    D = [np.zeros_like(th)]
    for k in range(1, nh + 1):
        V += [np.cos(k * th), np.sin(k * th)]
        D += [-k * np.sin(k * th), k * np.cos(k * th)]
    return np.array(V), np.array(D)


def ring_chain(A3, D3, G2, R, nquad=720, nh=NH):
    """Full VAM chain on the circle for wall law (A3,B=0,D3 | G2).  Returns diag6 =
    [EA, GA2, GA3, GJ, EI2, EI3] and the intermediate matrices."""
    th = (np.arange(nquad) + 0.5) * (2 * np.pi / nquad)
    wq = np.full(nquad, 2 * np.pi / nquad) * R          # ds = R dth
    V, Dh = eval_basis(th, nh)                          # (nb, nq)
    nb = V.shape[0]
    t2, t3 = -np.sin(th), np.cos(th)
    y2, y3 = -np.cos(th), -np.sin(th)
    x2, x3 = R * np.cos(th), R * np.sin(th)
    d2 = Dh / R                                         # contour derivative of basis

    nu = NF * nb                                        # field dofs
    nl = nb if NLAM else 0
    # index helper: field f, basis j -> f*nb + j
    def I(f, j):
        return f * nb + j

    # strain row builders: each returns (nrow, nu) H-part and L-part and (nrow,4) E-part
    Z = np.zeros
    # 6 classical rows + 2 shear rows, at each quad point: assemble as (8, nu/4) coeff
    # arrays evaluated per-theta, then integrate the energy.
    def rows_at():
        E = Z((8, 4, nquad)); H = Z((8, nu, nquad)); L = Z((8, nu, nquad))
        # e11
        E[0, 0] = 1.0; E[0, 2] = x3; E[0, 3] = -x2
        for j in range(nb):
            L[0, I(0, j)] = V[j]
        # e22
        for j in range(nb):
            H[1, I(1, j)] = t2 * d2[j]
            H[1, I(2, j)] = t3 * d2[j]
        # 2e12
        E[2, 1] = R
        for j in range(nb):
            H[2, I(0, j)] = d2[j]
            L[2, I(1, j)] = t2 * V[j]
            L[2, I(2, j)] = t3 * V[j]
        # k11
        E[3, 2] = t2; E[3, 3] = t3
        for j in range(nb):
            L[3, I(4, j)] = t2 * V[j]
            L[3, I(5, j)] = t3 * V[j]
        # k22
        for j in range(nb):
            H[4, I(3, j)] = -d2[j]
        # k12
        E[5, 1] = -1.0
        for j in range(nb):
            L[5, I(3, j)] = -V[j]
            H[5, I(4, j)] = t2 * d2[j]
            H[5, I(5, j)] = t3 * d2[j]
        # 2g13
        for j in range(nb):
            L[6, I(1, j)] = y2 * V[j]
            L[6, I(2, j)] = y3 * V[j]
            H[6, I(4, j)] = t2 * V[j]
            H[6, I(5, j)] = t3 * V[j]
        # 2g23
        for j in range(nb):
            H[7, I(1, j)] = y2 * d2[j]
            H[7, I(2, j)] = y3 * d2[j]
            H[7, I(3, j)] = -V[j]
        return E, H, L

    E, H, L = rows_at()
    K8 = np.zeros((8, 8))
    K8[:3, :3] = A3; K8[3:6, 3:6] = D3; K8[6:, 6:] = G2

    def bil(P, Q_):
        """energy bilinear  int P^T K8 Q dA  -> (dimP, dimQ)"""
        KQ = np.einsum("rs,sqn->rqn", K8, Q_)
        return np.einsum("rpn,rqn,n->pq", P, KQ, wq)

    Dhh = bil(H, H); Dhe = bil(H, E); Dee = bil(E, E)
    Dhl = bil(H, L); Dll = bil(L, L); Dle = bil(L, E)

    # drilling constraint rows: g_e (4), g_h (nu), g_l (nu) per theta
    ge = Z((4, nquad)); gh = Z((nu, nquad)); gl = Z((nu, nquad))
    ge[1] = -R / 2.0
    for j in range(nb):
        gh[I(0, j)] = 0.5 * d2[j]
        gh[I(4, j)] = y2 * V[j]
        gh[I(5, j)] = y3 * V[j]
        gl[I(1, j)] = -0.5 * t2 * V[j]
        gl[I(2, j)] = -0.5 * t3 * V[j]
    # multiplier basis = same trig basis
    Gh = np.einsum("jn,pn,n->jp", V, gh, wq)            # (nl, nu)
    Ge_ = np.einsum("jn,pn,n->jp", V, ge, wq)           # (nl, 4)
    Gl = np.einsum("jn,pn,n->jp", V, gl, wq)

    # kernel constraints <w1>=<w2>=<w3>=<om1>=0  (rigid modes)
    Cker = Z((4, nu))
    for f, r in ((0, 0), (1, 1), (2, 2), (3, 3)):
        Cker[r, I(f, 0)] = 2 * np.pi * R                # only the n=0 cosine integrates
    # KKT: [Dhh Gh^T Cker^T; Gh 0 0; Cker 0 0]
    n1 = nu; n2 = n1 + nl; n3 = n2 + 4
    KK = Z((n3, n3))
    KK[:n1, :n1] = Dhh
    KK[:n1, n1:n2] = Gh.T; KK[n1:n2, :n1] = Gh
    KK[:n1, n2:] = Cker.T; KK[n2:, :n1] = Cker

    def solve(rhs_u, rhs_l=None):
        r = Z((n3, rhs_u.shape[1]))
        r[:n1] = rhs_u
        if rhs_l is not None:
            r[n1:n2] = rhs_l
        sol = np.linalg.lstsq(KK, r, rcond=None)[0]
        return sol[:n1]

    V0 = solve(-Dhe, -Ge_)                               # zeroth (KKT with g = 0)
    Abeam = V0.T @ Dhe + Dee
    Abeam = 0.5 * (Abeam + Abeam.T)
    # first order (paper eq:first, with the constraint rows on the l-side: g_l V0' + g_h V1 = 0)
    rhs = -((Dhl - Dhl.T) @ V0 - Dle)
    V1 = solve(rhs, -(Gl @ V0))     # weak first-order constraint g_h V1 + g_l V0 = 0
    Bb = V0.T @ Dhl @ V0 + Dle.T @ V0
    Ds = (Dhl + Dhl.T) @ V0 + Dle
    Cc = V1.T @ Ds + V0.T @ Dll @ V0
    Cc = 0.5 * (Cc + Cc.T)

    Ainv = np.linalg.inv(Abeam)
    for Q in (np.array([[0, 0], [0, 0], [0, -1.0], [1.0, 0]]),
              np.array([[0, 0], [0, 0], [0, 1.0], [-1.0, 0]]),
              np.array([[0, 0], [0, 0], [1.0, 0], [0, 1.0]])):
        M = Q.T @ Ainv @ (Cc - Bb.T @ Ainv @ Bb) @ Ainv @ Q
        if np.linalg.det(M) > 0:
            Gb = np.linalg.inv(M)
            if np.all(np.diag(Gb) > 0):
                break
    Y = Bb.T @ Ainv @ Q @ Gb
    X = Abeam + Y @ np.linalg.inv(Gb) @ Y.T
    diag6 = np.array([X[0, 0], Gb[0, 0], Gb[1, 1], X[1, 1], X[2, 2], X[3, 3]])
    return diag6, dict(A=Abeam, B=Bb, C=Cc, G=Gb, X=X, Q=Q)


# ---------------- wall laws ----------------
def wall_iso(E, nu, h):
    q = E / (1 - nu ** 2)
    A3 = q * h * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    D3 = q * h ** 3 / 12 * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    G = E / (2 * (1 + nu))
    G2 = (5.0 / 6.0) * G * h * np.eye(2)
    return A3, D3, G2


def wall_ply45(h):
    """single [-45] ply of the paper's tube material via the code's own ABD builder."""
    from emit_abd import material_db_from_yaml
    from msg_rm_plate import rm_plate_msg
    mdb = {"ud": {"E": [37e9, 9e9, 9e9], "G": [4e9, 4e9, 4e9],
                  "nu": [0.28, 0.28, 0.28], "rho": 0.0}}
    r = rm_plate_msg([h], [-45.0], ["ud"], mdb, fraction=0.5)
    A8 = np.asarray(r["A6"])
    return A8[:3, :3], A8[3:, 3:], np.asarray(r["G_msg"])


# ---------------- PHASE A: validate vs ring_indep ----------------
def ring_code(A3, D3, G2, R, N=720):
    from run_ring_indep import ring_indep
    th = np.arange(N) * 2 * np.pi / N
    rx = np.zeros((N, 3)); rx[:, 0] = R * np.cos(th); rx[:, 1] = R * np.sin(th)
    cells = np.column_stack([np.arange(N), (np.arange(N) + 1) % N])
    re3 = np.zeros((N, 3))
    mid = 0.5 * (rx[cells[:, 0]] + rx[cells[:, 1]])
    re3[:, 0] = -mid[:, 0]; re3[:, 1] = -mid[:, 1]
    re3 /= np.linalg.norm(re3, axis=1)[:, None]
    D6 = np.zeros((6, 6)); D6[:3, :3] = A3; D6[3:, 3:] = D3
    D_by = [D6]; G_by = [G2]
    rsub = np.zeros(N, int)
    from segment_element import compute_k22
    ori = np.zeros((N, 9)); ori[:, 3:6] = np.column_stack([-np.sin(np.arctan2(mid[:, 1], mid[:, 0])),
                                                           np.cos(np.arctan2(mid[:, 1], mid[:, 0])),
                                                           np.zeros(N)])
    ori[:, 6:9] = re3
    k22 = compute_k22(mid, ori[:, 3:6], re3, cells)
    C = ring_indep(rx, cells, rsub, re3, D_by, G_by, k22, 2, [0, 1], shear="mitc4_g23",
                   lam_space="elem")
    C = 0.5 * (C + C.T)
    return np.diag(C)


if __name__ == "__main__":
    LBL = ["EA  ", "GA2 ", "GA3 ", "GJ  ", "EI2 ", "EI3 "]
    R = 0.0715
    for name, wall in (("iso E=70GPa nu=0.3", wall_iso(70e9, 0.3, R / 2)),
                       ("iso thin", wall_iso(70e9, 0.3, R / 10)),
                       ("[-45] R/h=2", wall_ply45(R / 2)),
                       ("[-45] R/h=10", wall_ply45(R / 10))):
        A3, D3, G2 = wall
        d_an, info = ring_chain(A3, D3, G2, R)
        d_cd = ring_code(A3, D3, G2, R)
        print("== %s ==" % name)
        for k in range(6):
            print("  %s analytic %.6e  code %.6e  diff %+7.3f%%"
                  % (LBL[k], d_an[k], d_cd[k], 100 * (d_an[k] - d_cd[k]) / d_cd[k]))
