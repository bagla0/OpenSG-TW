"""fsm_buckling.py -- semi-analytical Finite Strip Method for LOCAL/distortional buckling of a
prismatic thin-walled section from cross-section data only (contour + per-strip laminate ABD +
pre-buckling membrane resultants N).  The eigenvalue problem is posed on the 1-D cross-section with an
ANALYTIC sinusoidal longitudinal variation of half-wavelength a; sweep a -> signature curve.

This is the cost-justified RM-OpenSG buckling kernel: it consumes exactly {contour, ABD, N} that the
homogenization + dehomogenization already produce, and reduces a full 3-D shell eigenproblem to a tiny
(4 x n_contour)-DOF problem per half-wavelength.

Strip: flat laminated plate strip, width b, single half-wave, k=pi/a.  Longitudinal shapes:
u ~ cos(kx) (warping), v,w,theta ~ sin(kx).  Transverse: linear (u,v), cubic Hermite (w,theta).
Local DOF per node [u, v, w, theta=dw/dy]; 8 per strip.  (Single-harmonic -> the anisotropic 16/26 ABD
terms integrate out; use multiharmonic for full anisotropy -- see module note.)

Validation target: axially-compressed isotropic cylinder, classical N_cr = E t^2 / (R sqrt(3(1-nu^2)))."""
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

G2 = [(-1 / np.sqrt(3), 1.0), (1 / np.sqrt(3), 1.0)]


def _hermite(y, b):
    xi = y / b
    H = np.array([1 - 3 * xi**2 + 2 * xi**3, b * (xi - 2 * xi**2 + xi**3),
                  3 * xi**2 - 2 * xi**3, b * (-xi**2 + xi**3)])
    dH = np.array([(-6 * xi + 6 * xi**2) / b, 1 - 4 * xi + 3 * xi**2,
                   (6 * xi - 6 * xi**2) / b, -2 * xi + 3 * xi**2])
    ddH = np.array([(-6 + 12 * xi) / b**2, (-4 + 6 * xi) / b,
                    (6 - 12 * xi) / b**2, (-2 + 6 * xi) / b])
    return H, dH, ddH


# Include the Nxy term in the membrane geometric stiffness (solve_fsm_multi / solve_fsm_connected).
# Set FSM_NO_NXY=1 to reproduce the previous diag(Nx,Ny) behaviour for A/B comparison.
INCLUDE_NXY = not bool(int(os.environ.get("FSM_NO_NXY", "0")))


def strip_ke_kg(b, a, ABD, N, full_geom=False):
    """local 8x8 elastic Ke and geometric Kg for a strip.  DOF order [u1,v1,w1,th1, u2,v2,w2,th2].
    ABD 6x6 [[A,B],[B,D]] Voigt [11,22,12]; N=[Nx,Ny,Nxy] (Nx=longitudinal/axial).
    full_geom=False -> classic FSM w-only geometric stiffness (avoids the long-a warping mechanism)."""
    k = np.pi / a
    A = ABD[:3, :3].copy(); B = ABD[:3, 3:].copy(); D = ABD[3:, 3:].copy()
    for M in (A, B, D):                                        # single harmonic: 16/26 terms integrate to 0
        M[0, 2] = M[2, 0] = M[1, 2] = M[2, 1] = 0.0
    Nx, Ny, Nxy = N
    Nm = np.array([[Nx, 0.0], [0.0, Ny]])                     # Nxy cross-term also vanishes (sin*cos)
    Ke = np.zeros((8, 8)); Kg = np.zeros((8, 8))
    for xg, wg in G2:
        y = (xg + 1) / 2 * b; Jw = b / 2 * wg
        N1 = 1 - y / b; N2 = y / b; dN1 = -1.0 / b; dN2 = 1.0 / b
        H, dH, ddH = _hermite(y, b)
        Bm = np.zeros((3, 8)); Bb = np.zeros((3, 8))
        Bm[0, 0] = -k * N1; Bm[0, 4] = -k * N2                # eps_x = u,x
        Bm[1, 1] = dN1;     Bm[1, 5] = dN2                    # eps_y = v,y
        Bm[2, 0] = dN1;     Bm[2, 4] = dN2                    # gam_xy = u,y ...
        Bm[2, 1] = k * N1;  Bm[2, 5] = k * N2                 #        ... + v,x
        wi = [2, 3, 6, 7]                                     # w,theta DOF indices
        for c, idx in enumerate(wi):
            Bb[0, idx] = k * k * H[c]                         # kap_x = -w,xx
            Bb[1, idx] = -ddH[c]                              # kap_y = -w,yy
            Bb[2, idx] = -2 * k * dH[c]                       # kap_xy = -2 w,xy
        Ke += (a / 2) * Jw * (Bm.T @ A @ Bm + Bm.T @ B @ Bb + Bb.T @ B.T @ Bm + Bb.T @ D @ Bb)
        # geometric: gradients of u,v,w (curvature captured by tilted-strip assembly, as in the shell KG)
        Gu = np.zeros((2, 8)); Gv = np.zeros((2, 8)); Gw = np.zeros((2, 8))
        Gu[0, 0] = -k * N1; Gu[0, 4] = -k * N2; Gu[1, 0] = dN1; Gu[1, 4] = dN2      # u,x ; u,y
        Gv[0, 1] = k * N1;  Gv[0, 5] = k * N2;  Gv[1, 1] = dN1; Gv[1, 5] = dN2      # v,x ; v,y
        for c, idx in enumerate(wi):
            Gw[0, idx] = k * H[c]; Gw[1, idx] = dH[c]                               # w,x ; w,y
        Kg += (a / 2) * Jw * Gw.T @ Nm @ Gw
        if full_geom:
            Kg += (a / 2) * Jw * (Gu.T @ Nm @ Gu + Gv.T @ Nm @ Gv)
    return Ke, Kg


def _strip_B(b, k):
    """Gauss-point amplitude matrices for a strip at wavenumber k (longitudinal factor NOT included).
    Bsin (4x8) = [eps_x, eps_y, kap_x, kap_y] (sin-type); Bcos (2x8) = [gam_xy, kap_xy] (cos-type);
    Gw (2x8) = [w,x ; w,y]. DOF [u1,v1,w1,th1, u2,v2,w2,th2]."""
    out = []
    wi = [2, 3, 6, 7]
    for xg, wg in G2:
        y = (xg + 1) / 2 * b; Jw = b / 2 * wg
        N1 = 1 - y / b; N2 = y / b; dN1 = -1.0 / b; dN2 = 1.0 / b
        H, dH, ddH = _hermite(y, b)
        Bsin = np.zeros((4, 8)); Bcos = np.zeros((2, 8)); Gw = np.zeros((2, 8))
        Bsin[0, 0] = -k * N1; Bsin[0, 4] = -k * N2                # eps_x
        Bsin[1, 1] = dN1;     Bsin[1, 5] = dN2                    # eps_y
        Bcos[0, 0] = dN1; Bcos[0, 4] = dN2; Bcos[0, 1] = k * N1; Bcos[0, 5] = k * N2   # gam_xy
        for c, idx in enumerate(wi):
            Bsin[2, idx] = k * k * H[c]; Bsin[3, idx] = -ddH[c]   # kap_x, kap_y
            Bcos[1, idx] = -2 * k * dH[c]                         # kap_xy
            Gw[0, idx] = k * H[c]; Gw[1, idx] = dH[c]             # w,x ; w,y
        out.append((Bsin, Bcos, Gw, Jw))
    return out


def solve_fsm_multi(nodes2d, strips, ABD_s, N_s, L, M, n_modes=4, return_vecs=False):
    """MULTI-HARMONIC FSM: member of length L, harmonics m=1..M coupled through the anisotropic 16/26 ABD
    terms.  Returns the lowest buckling factors on N.  (Single-harmonic = M=1 special case but keeps only
    the orthotropic core; the coupling appears only for M>=2 via opposite-parity harmonics.)"""
    nn = len(nodes2d); nd = 4 * nn; ndof = M * nd
    ss = [0, 1, 3, 4]; cc = [2, 5]
    cmm = lambda m, mp: (2 * L / np.pi) * m / (m * m - mp * mp) if (m + mp) % 2 == 1 else 0.0
    rk, ck, kv = [], [], []; rg, cg, kgv = [], [], []

    def add(dr, dc, dv, block_m, block_mp, i, j, Mat):
        gd = np.r_[4 * i:4 * i + 4, 4 * j:4 * j + 4]
        gi = block_m * nd + gd; gj = block_mp * nd + gd
        for p in range(8):
            for q in range(8):
                dr.append(gi[p]); dc.append(gj[q]); dv.append(Mat[p, q])

    for e, (i, j) in enumerate(strips):
        P1, P2 = nodes2d[i], nodes2d[j]; d = P2 - P1; b = np.linalg.norm(d); cy, cz = d / b
        Tn = np.array([[1, 0, 0, 0], [0, cy, cz, 0], [0, -cz, cy, 0], [0, 0, 0, 1.0]])
        T = np.zeros((8, 8)); T[:4, :4] = Tn; T[4:, 4:] = Tn
        ABD = ABD_s[e]; Ass = ABD[np.ix_(ss, ss)]; Acc = ABD[np.ix_(cc, cc)]; Asc = ABD[np.ix_(ss, cc)]
        Nx, Ny, Nxy = N_s[e]; Nm = np.array([[Nx, 0.0], [0.0, Ny]])
        Bk = {m: _strip_B(b, m * np.pi / L) for m in range(1, M + 1)}
        for m in range(1, M + 1):
            Ke = np.zeros((8, 8)); Kge = np.zeros((8, 8))
            for Bsin, Bcos, Gw, Jw in Bk[m]:
                Ke += (L / 2) * Jw * (Bsin.T @ Ass @ Bsin + Bcos.T @ Acc @ Bcos)
                Kge += (L / 2) * Jw * (Gw.T @ Nm @ Gw)
            add(rk, ck, kv, m - 1, m - 1, i, j, T.T @ Ke @ T); add(rg, cg, kgv, m - 1, m - 1, i, j, T.T @ Kge @ T)
        for m in range(1, M + 1):
            for mp in range(m + 1, M + 1):
                if (m + mp) % 2 == 0:
                    continue
                Kc = np.zeros((8, 8))
                for (Bs_m, _, _, Jw), (_, Bc_mp, _, _) in zip(Bk[m], Bk[mp]):
                    Kc += cmm(m, mp) * Jw * (Bs_m.T @ Asc @ Bc_mp)
                for (_, Bc_m, _, Jw), (Bs_mp, _, _, _) in zip(Bk[m], Bk[mp]):
                    Kc += cmm(mp, m) * Jw * (Bc_m.T @ Asc.T @ Bs_mp)
                KcG = T.T @ Kc @ T
                add(rk, ck, kv, m - 1, mp - 1, i, j, KcG); add(rk, ck, kv, mp - 1, m - 1, i, j, KcG.T)
                # FULL membrane geometric stiffness: w' [[Nx,Nxy],[Nxy,Ny]] w'.  The Nxy cross term pairs
                # w,x (cos-type) with w,y (sin-type), so it carries the SAME sin-cos integral cmm that
                # justifies the Asc elastic coupling above.  Keeping Asc while dropping Nxy is inconsistent:
                # for an off-axis laminate A16 != 0, so an axial load generates Nxy and the anisotropy's
                # contribution to the GEOMETRIC stiffness would otherwise be lost even though it is present
                # in the elastic one.  Vanishes identically for m == m' and whenever Nxy == 0 (isotropic
                # under axial load), so this is a no-op on every previously verified isotropic result.
                if INCLUDE_NXY and Nxy != 0.0:
                    KgC = np.zeros((8, 8))
                    for (_, _, Gw_m, Jw), (_, _, Gw_mp, _) in zip(Bk[m], Bk[mp]):
                        KgC += Jw * Nxy * (cmm(mp, m) * np.outer(Gw_m[0], Gw_mp[1])
                                           + cmm(m, mp) * np.outer(Gw_m[1], Gw_mp[0]))
                    KgCg = T.T @ KgC @ T
                    add(rg, cg, kgv, m - 1, mp - 1, i, j, KgCg)
                    add(rg, cg, kgv, mp - 1, m - 1, i, j, KgCg.T)
    K = sp.coo_matrix((kv, (rk, ck)), (ndof, ndof)).tocsc()
    Kg = sp.coo_matrix((kgv, (rg, cg)), (ndof, ndof)).tocsc()
    w, V = eigsh(-Kg, k=n_modes, M=K, which="LA")
    keep = w > 1e-9; w = w[keep]; V = V[:, keep]
    if len(w) == 0:
        return (np.array([np.inf]), None) if return_vecs else np.array([np.inf])
    order = np.argsort(1.0 / w); lam = (1.0 / w)[order]; V = V[:, order]
    if return_vecs:
        return lam, V.reshape(M, nn, 4, -1)               # (M harmonics, nn nodes, [u,Dy,Dz,th], modes)
    return lam


def solve_fsm_connected(sections, L, M, n_modes=6, ngauss=48, return_vecs=False, strips=None):
    """CONNECTED multi-section FSM: all supplied cross-sections contribute to ONE eigenproblem.

    The per-station method solves each cross-section as an isolated PRISMATIC member and takes the minimum.
    That is structurally wrong whenever the buckle SPANS the taper: measured on the tapered square, the
    3-D solid's mode peaks at x~0.5-0.7 and is spread over x~0.2-1.4, so no single prismatic section
    represents it and the min-over-stations rule picks the (weakest, but unreachable) root -> ~22% low.

    Formulation.  Keep the harmonic series over the WHOLE member,
        u = sum_m U_m(xi) cos(k_m x),   v,w = sum_m {V_m,W_m}(xi) sin(k_m x),   k_m = m pi / L,
    with the contour parametrized by normalized arc length xi so node j corresponds across sections, but
    let the GEOMETRY, ABD and pre-stress N vary with x (interpolated between the supplied sections) and do
    the longitudinal integral NUMERICALLY.  For a prismatic member the trig products integrate to L/2 * delta_mm'
    and this reduces exactly to solve_fsm_multi; when the section varies, that orthogonality BREAKS, the
    harmonics couple, and a superposition can localize the buckle anywhere along the span -- which is what
    lets the model feel the wall narrowing inside one buckle.

    Row trig types (needed for the numeric x-integration):
        Bsin rows [eps_x, eps_y, kap_x, kap_y] ~ sin(k_m x)     Bcos rows [gam_xy, kap_xy] ~ cos(k_m x)
        Gw row0 = w,x ~ cos(k_m x)                              Gw row1 = w,y ~ sin(k_m x)

    sections : list of (x_k, nodes2d_k (nn,2), ABD_k (list per strip), N_k (list per strip)), x ascending.
               Sections must share the same nn and the same node ordering (arc-length correspondence).
    strips   : optional (ns,2) int connectivity, 0-based, SHARED by every section (node correspondence is
               already mandatory, so one topology serves all x).  Default None = a single closed ring
               [(i, i+1 mod nn)], which is what a plain tube needs.  Pass the station's own connectivity to
               connect a BRANCHED contour -- a blade cross-section is multi-cell with shear webs, and the
               default ring cannot represent it (the webs would simply be dropped).  A web that vanishes
               across the span degrades gracefully: its strip length goes to zero and the `b <= 0` guard
               below skips it.
    Returns lam (ascending |.|), or (lam, V) with V shaped (M, nn, 4, n_modes) if return_vecs.
    """
    xs_k = np.array([s[0] for s in sections], float)
    P_k = np.array([np.asarray(s[1], float) for s in sections])          # (K, nn, 2)
    A_k = np.array([np.asarray(s[2], float) for s in sections])          # (K, ns, 6, 6)
    N_k = np.array([np.asarray(s[3], float) for s in sections])          # (K, ns, 3)
    nn = P_k.shape[1]
    if strips is None:
        strips = np.array([[i, (i + 1) % nn] for i in range(nn)])     # single closed ring: no webs
    else:
        strips = np.asarray(strips, int)
        if strips.ndim != 2 or strips.shape[1] != 2:
            raise ValueError("strips must be (ns,2) node-index pairs, got shape %s" % (strips.shape,))
        if strips.min() < 0 or strips.max() >= nn:
            raise ValueError("strips index outside [0,%d): min=%d max=%d — 0-based indices required"
                             % (nn, strips.min(), strips.max()))
    ns = len(strips)
    if A_k.shape[1] != ns or N_k.shape[1] != ns:
        raise ValueError("ABD/N must carry one entry per strip: ns=%d but ABD has %d and N has %d"
                         % (ns, A_k.shape[1], N_k.shape[1]))
    nd = 4 * nn; ndof = M * nd
    ss = [0, 1, 3, 4]; cc = [2, 5]

    def interp(x):
        """linear interpolation of section geometry / ABD / N at position x (this is the 'connection')."""
        j = int(np.clip(np.searchsorted(xs_k, x) - 1, 0, len(xs_k) - 2))
        w = (x - xs_k[j]) / (xs_k[j + 1] - xs_k[j]); w = min(max(w, 0.0), 1.0)
        return ((1 - w) * P_k[j] + w * P_k[j + 1],
                (1 - w) * A_k[j] + w * A_k[j + 1],
                (1 - w) * N_k[j] + w * N_k[j + 1])

    gx, gw = np.polynomial.legendre.leggauss(ngauss)
    gx = 0.5 * L * (gx + 1.0); gw = 0.5 * L * gw
    K = np.zeros((ndof, ndof)); KG = np.zeros((ndof, ndof))
    kms = np.array([m * np.pi / L for m in range(1, M + 1)])

    for xg, wg in zip(gx, gw):
        P, Aall, Nall = interp(xg)
        sm = np.sin(kms * xg); cm = np.cos(kms * xg)
        for e in range(ns):
            i, j = strips[e]
            d = P[j] - P[i]; b = float(np.linalg.norm(d))
            if b <= 0:
                continue
            cy, cz = d / b
            Tn = np.array([[1, 0, 0, 0], [0, cy, cz, 0], [0, -cz, cy, 0], [0, 0, 0, 1.0]])
            T = np.zeros((8, 8)); T[:4, :4] = Tn; T[4:, 4:] = Tn
            ABD = Aall[e]
            Ass = ABD[np.ix_(ss, ss)]; Acc = ABD[np.ix_(cc, cc)]; Asc = ABD[np.ix_(ss, cc)]
            Nx, Ny, Nxy = Nall[e]
            Bk = [_strip_B(b, km) for km in kms]
            gd = np.r_[4 * i:4 * i + 4, 4 * j:4 * j + 4]
            for mi in range(M):
                for mj in range(M):
                    Ke = np.zeros((8, 8)); Kge = np.zeros((8, 8))
                    for (Bs_i, Bc_i, Gw_i, Jw), (Bs_j, Bc_j, Gw_j, _) in zip(Bk[mi], Bk[mj]):
                        Ke += Jw * (Bs_i.T @ Ass @ Bs_j * sm[mi] * sm[mj]
                                    + Bs_i.T @ Asc @ Bc_j * sm[mi] * cm[mj]
                                    + Bc_i.T @ Asc.T @ Bs_j * cm[mi] * sm[mj]
                                    + Bc_i.T @ Acc @ Bc_j * cm[mi] * cm[mj])
                        Kge += Jw * (Nx * np.outer(Gw_i[0], Gw_j[0]) * cm[mi] * cm[mj]
                                     + Ny * np.outer(Gw_i[1], Gw_j[1]) * sm[mi] * sm[mj])
                        # FULL membrane geometric stiffness: the Nxy cross term pairs w,x (cos) with w,y
                        # (sin).  Here the x-integral is NUMERICAL over a varying section, so the cos-sin
                        # product does NOT vanish -- exactly as for the four elastic trig products retained
                        # just above.  Dropping it while keeping the elastic sin-cos coupling is
                        # inconsistent.  Identically zero when Nxy == 0, so isotropic axial results are
                        # unchanged.
                        if INCLUDE_NXY and Nxy != 0.0:
                            Kge += Jw * Nxy * (np.outer(Gw_i[0], Gw_j[1]) * cm[mi] * sm[mj]
                                               + np.outer(Gw_i[1], Gw_j[0]) * sm[mi] * cm[mj])
                    KeG = T.T @ Ke @ T * wg; KgG = T.T @ Kge @ T * wg
                    gi = mi * nd + gd; gj_ = mj * nd + gd
                    K[np.ix_(gi, gj_)] += KeG; KG[np.ix_(gi, gj_)] += KgG

    K = 0.5 * (K + K.T); KG = 0.5 * (KG + KG.T)
    Ks = sp.csc_matrix(K); KGs = sp.csc_matrix(KG)
    w, V = eigsh(-KGs, k=min(n_modes, ndof - 2), M=Ks, which="LA")
    keep = w > 1e-12; w = w[keep]; V = V[:, keep]
    if len(w) == 0:
        return (np.array([np.inf]), None) if return_vecs else np.array([np.inf])
    order = np.argsort(1.0 / w); lam = (1.0 / w)[order]; V = V[:, order]
    if return_vecs:
        return lam, V.reshape(M, nn, 4, -1)
    return lam


def solve_fsm(nodes2d, strips, ABD_s, N_s, a):
    """Assemble the closed/open section at half-wavelength a and return the buckling factors (on N).
    nodes2d (nn,2) cross-section coords (y,z); strips (ns,2) nodal-line pairs; ABD_s,N_s per strip.
    Global DOF per node [u(axial), Dy, Dz, theta(about axial)]; strip local (v,w) rotate by chord angle."""
    nn = len(nodes2d); ndof = 4 * nn
    rows, cols, kv, kgv = [], [], [], []
    for e, (i, j) in enumerate(strips):
        P1, P2 = nodes2d[i], nodes2d[j]
        d = P2 - P1; b = np.linalg.norm(d); cy, cz = d / b     # chord (local +y) direction
        Ke, Kg = strip_ke_kg(b, a, ABD_s[e], N_s[e])
        Tn = np.array([[1, 0, 0, 0], [0, cy, cz, 0], [0, -cz, cy, 0], [0, 0, 0, 1.0]])   # global->local per node
        T = np.zeros((8, 8)); T[:4, :4] = Tn; T[4:, 4:] = Tn
        KeG = T.T @ Ke @ T; KgG = T.T @ Kg @ T
        gd = np.r_[4 * i:4 * i + 4, 4 * j:4 * j + 4]
        for p in range(8):
            for q in range(8):
                rows.append(gd[p]); cols.append(gd[q]); kv.append(KeG[p, q]); kgv.append(KgG[p, q])
    K = sp.coo_matrix((kv, (rows, cols)), (ndof, ndof)).tocsc()
    Kg = sp.coo_matrix((kgv, (rows, cols)), (ndof, ndof)).tocsc()
    w, v = eigsh(-Kg, k=6, M=K, which="LA")
    keep = w > 1e-9; w = w[keep]; v = v[:, keep]
    if len(w) == 0:
        return np.array([np.inf]), None
    order = np.argsort(1.0 / w); lam = (1.0 / w)[order]
    return lam, v[:, order].reshape(nn, 4, -1)          # (nn nodes, [u,Dy,Dz,th], modes)


def signature_curve(nodes2d, strips, ABD_s, N_s, a_list):
    """buckling factor vs half-wavelength; returns (a_list, lam1(a)) and the critical (a*, lam*)."""
    lam1 = np.array([solve_fsm(nodes2d, strips, ABD_s, N_s, a)[0][0] for a in a_list])
    i = int(np.argmin(lam1))
    return np.asarray(a_list), lam1, a_list[i], lam1[i]


# ---------- laminate ABD (self-contained CLT) ----------
def clt_abd(plies, mat):
    """ABD 6x6 from plies [(angle_deg, thickness), ...] of an orthotropic mat {E1,E2,G12,nu12}, stacked
    OML->IML.  Reference = laminate mid-surface (center)."""
    E1, E2, G12, nu12 = mat["E1"], mat["E2"], mat["G12"], mat["nu12"]
    nu21 = nu12 * E2 / E1; den = 1 - nu12 * nu21
    Q = np.array([[E1 / den, nu12 * E2 / den, 0], [nu12 * E2 / den, E2 / den, 0], [0, 0, G12]])
    ttot = sum(t for _, t in plies)
    z = -ttot / 2
    A = np.zeros((3, 3)); B = np.zeros((3, 3)); D = np.zeros((3, 3))
    for ang, t in plies:
        c = np.cos(np.radians(ang)); s = np.sin(np.radians(ang))
        T = np.array([[c*c, s*s, 2*c*s], [s*s, c*c, -2*c*s], [-c*s, c*s, c*c - s*s]])
        Qb = np.linalg.inv(T) @ Q @ np.linalg.inv(T).T        # rotate to laminate axes
        z1 = z + t
        A += Qb * (z1 - z); B += Qb * (z1**2 - z**2) / 2; D += Qb * (z1**3 - z**3) / 3
        z = z1
    ABD = np.zeros((6, 6)); ABD[:3, :3] = A; ABD[:3, 3:] = B; ABD[3:, :3] = B; ABD[3:, 3:] = D
    return ABD


def iso_abd(E, nu, t):
    C = E / (1 - nu*nu)
    Qm = C * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    ABD = np.zeros((6, 6)); ABD[:3, :3] = Qm * t; ABD[3:, 3:] = Qm * t**3 / 12
    return ABD


def cyl_ring(R, nc):
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    nodes = np.column_stack([R * np.cos(th), R * np.sin(th)])
    strips = np.array([[i, (i + 1) % nc] for i in range(nc)])
    return nodes, strips


if __name__ == "__main__":
    # isotropic cylinder axial buckling: FSM signature curve vs classical
    E, nu, R, t = 200e9, 0.3, 1.0, 0.02
    Ncl = E * t**2 / (R * np.sqrt(3 * (1 - nu**2)))
    nc = 160
    nodes, strips = cyl_ring(R, nc)
    ABD = iso_abd(E, nu, t)
    ABD_s = [ABD] * len(strips); N_s = [np.array([-1.0, 0.0, 0.0])] * len(strips)   # unit axial compression
    a_list = np.geomspace(0.05, 3.0, 40)
    aa, lam1, ac, lc = signature_curve(nodes, strips, ABD_s, N_s, a_list)
    print("ISO cylinder FSM (nc=%d): classical N_cr=%.4e" % (nc, Ncl))
    print("  critical: a*=%.4f m  N_cr(FSM)=%.4e  ratio=%.4f" % (ac, lc, lc / Ncl))
    print("  signature (a, N_cr/Ncl):")
    for a, l in zip(aa[::4], lam1[::4]):
        print("    a=%.3f  N/Ncl=%.3f" % (a, l / Ncl))
