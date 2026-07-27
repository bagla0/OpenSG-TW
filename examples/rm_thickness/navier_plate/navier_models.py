"""navier_models.py -- the two plate-level models for the double-sine loaded plate.

Both are single-harmonic Navier solutions of the same 5-dof plate problem
(u0, v0, w0, psi_x, psi_y), differing only in the transverse-shear stiffness and in the
through-thickness recovery:

  fsdt_mr   FSDT with k = 5/6  +  the Mendonca-Ruviaro recovery chain (FEAD 260 (2026)
            104609, Eqs. 14-44): equilibrium integration for tau_xz, tau_yz, sigma_z
            (with the multiplicative top-face scaling, their Eq. 32), eps_z from the 3-D
            compliance -> w by integration (shift, Eq. 35), gamma_xz from the 3-D
            compliance -> u by integration (shift + rotation, Eqs. 41-44).  This is the
            ANALYTIC version of their process -- exactly the red-dashed curves of their
            figures, which their FE + sequential smoothing converges to.
  msg_mr    OpenSG-RM: the same Navier problem closed with the MSG transverse-shear
            stiffness G_MSG, and every through-thickness field taken DIRECTLY from the
            structure-gene warping -- in-plane stress from the zeroth-order warping,
            tau_xz/tau_yz from the first-order gradient warping, sigma_z from
            through-thickness equilibrium, and u(z), w(z) as plate kinematics PLUS the
            recovered warping displacement.  No correction, shift, rotation or scaling
            step exists in this chain.

Kinematics and sign conventions follow the paper: u = u0 + z psi_x, gamma_xz = w,x+psi_x.

Amplitude families (m = n = 1):
  u, tau_xz : cos(px) sin(qy)   -> extracted at A = (0, a/2)
  v, tau_yz : sin(px) cos(qy)   -> extracted at C = (a/2, 0)
  w, sigma_{x,y,z} : sin(px) sin(qy)  -> extracted at B = (a/2, a/2)
  gamma_xy, tau_xy : cos(px) cos(qy)

Normalisations (their Eq. 52):
  tbar_xz = (H/(q11 a)) tau_xz,   sbar_z = sigma_z/q11,
  ubar = (H^2 E2/(q11 a^3)) u,    wbar = (100 H^3 E2/(q11 a^4)) w.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

from jaxcfg import jax, jnp                       # noqa: E402
import sg_plate as SG                             # noqa: E402
from models import plane_stress_C                 # noqa: E402
from exact_navier import ExactNavier              # noqa: E402

# ------------------------------------------------------------------ material data
# Mendonca & Ruviaro sec. 5 (Pagano graphite/epoxy + the Pagano-1970 sandwich core).
MATDB_MR = {
    'mr_lam': {'E': [172.25e9, 6.89e9, 6.89e9],
               'G': [3.445e9, 3.445e9, 1.387e9],
               'nu': [0.25, 0.25, 0.25], 'rho': 1.0},
    'mr_face': {'E': [172.25e9, 6.89e9, 6.89e9],
                'G': [3.445e9, 3.445e9, 1.378e9],
                'nu': [0.25, 0.25, 0.25], 'rho': 1.0},
    'mr_core': {'E': [275.6e6, 275.6e6, 3445.0e6],
                'G': [110.24e6, 413.4e6, 413.4e6],
                'nu': [0.25, 0.02, 0.02], 'rho': 1.0},
}
E2_REF = 6.89e9        # "elastic modulus on principal orthotropic direction 2"

CASES = {
    'sym':      (lambda H: ([H / 3] * 3, [0., 90., 0.],
                            ['mr_lam'] * 3)),
    'asym':     (lambda H: ([H / 2] * 2, [0., 90.],
                            ['mr_lam'] * 2)),
    'sandwich': (lambda H: ([0.1 * H, 0.8 * H, 0.1 * H], [0., 0., 0.],
                            ['mr_face', 'mr_core', 'mr_face'])),
}

SS = np.array([1., 1., 0., 1., 1., 0.])      # rows of E varying as sin*sin
CC = np.array([0., 0., 1., 0., 0., 1.])      # rows varying as cos*cos


# ------------------------------------------------------------------ Navier solve
def navier_matrices(p, q):
    """L (6x5): d=[U0,V0,W0,Px,Py] -> plate strain amplitudes; Lg (2x5) -> [gxz,gyz]."""
    L = np.zeros((6, 5))
    L[0, 0] = -p                      # eps0_x
    L[1, 1] = -q                      # eps0_y
    L[2, 0] = q; L[2, 1] = p          # gamma0_xy   (CC)
    L[3, 3] = -p                      # kappa_x
    L[4, 4] = -q                      # kappa_y
    L[5, 3] = q; L[5, 4] = p          # kappa_xy    (CC)
    Lg = np.zeros((2, 5))
    Lg[0, 2] = p; Lg[0, 3] = 1.0      # gamma_xz = p W0 + Px   (CS)
    Lg[1, 2] = q; Lg[1, 4] = 1.0      # gamma_yz = q W0 + Py   (SC)
    return L, Lg


def navier_solve(A6, G2, p, q, q11):
    """Solve the 5x5 single-harmonic problem.  Returns (d, Ehat, ghat)."""
    L, Lg = navier_matrices(p, q)
    A6 = np.asarray(A6); G2 = np.asarray(G2)
    K = L.T @ A6 @ L + Lg.T @ G2 @ Lg
    f = np.array([0.0, 0.0, q11, 0.0, 0.0])
    d = np.linalg.solve(K, f)
    return d, L @ d, Lg @ d


def point_args(Ehat, p, q):
    """(E6, dE1, dE2) at the three extraction points A=(0,a/2), B=(a/2,a/2), C=(a/2,0)."""
    Ehat = np.asarray(Ehat)
    A = (np.zeros(6), p * SS * Ehat, -q * CC * Ehat)
    B = (SS * Ehat, np.zeros(6), np.zeros(6))
    C = (np.zeros(6), -p * CC * Ehat, q * SS * Ehat)
    return A, B, C


def _cumtrapz(y, z):
    y = np.asarray(y); z = np.asarray(z)
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(z))
    return out


# ---------------------------------------------------------------------- OpenSG-RM
def msg_mr(sg, p, q, q11, n_out=41):
    """OpenSG-RM: Navier with G_MSG + direct MSG recovery.  All amplitudes.

    FRAME NOTE (two lessons learnt numerically, kept because they cost a day each):
    (1) STRESS drivers must be the MOMENT-CONSISTENT RM measures.  The classical
        curvature p^2 W0 of the total deflection contains the shear deflection, which
        carries no moment; at a/H = 4 with E1/G13 = 50 the shear deflection dominates
        W0, so classical-measure drivers overdrive the bending stress ~5x and the
        recovered tau_xz with them.
    (2) The DISPLACEMENT warping must be taken as the pure through-thickness DEVIATION:
        its least-squares linear-in-z part is removed, because the RM director psi_x
        already carries the mean shear rotation -- keeping both double-counts it.
        This is the same rotation gauge Mendonca-Ruviaro impose on their recovered u
        (their Eqs. 43-44), which makes the comparison like-for-like.
    """
    d, Ehat, ghat = navier_solve(sg['A6'], sg['G_msg'], p, q, q11)
    U0, V0, W0, Px, Py = d
    argA, argB, argC = point_args(Ehat, p, q)

    z, _, SigB, warpB = SG.recover(sg, *argB, n_per_layer_out=n_out, return_warp=True)
    _, _, SigA, warpA = SG.recover(sg, *argA, n_per_layer_out=n_out, return_warp=True)
    _, _, SigC = SG.recover(sg, *argC, n_per_layer_out=n_out)

    txz = np.asarray(SigA[:, 4])                 # CS amplitude at A
    tyz = np.asarray(SigC[:, 3])                 # SC amplitude at C
    sz = _cumtrapz(p * txz + q * tyz, z)         # SS amplitude (0 at the bottom face)

    def _dev(wcomp):
        """remove the least-squares linear-in-z part (mean is already ~0)."""
        wcomp = np.asarray(wcomp)
        s = np.trapezoid(z * wcomp, z) / np.trapezoid(z * z, z)
        return wcomp - s * z

    u = U0 + z * Px + _dev(warpA[:, 0])          # RM line + zig-zag deviation
    w = W0 + np.asarray(warpB[:, 2])
    return dict(z=z, d=d, txz=txz, tyz=tyz, sz=sz, u=u, w=w,
                sx=np.asarray(SigB[:, 0]), sy=np.asarray(SigB[:, 1]),
                sz_top_closure=abs(sz[-1] / q11 - 1.0))


# ------------------------------------------------------- Mendonca-Ruviaro process
def fsdt_shear2(sg, ks=5.0 / 6.0):
    C = np.asarray(sg['C_layers'])
    t = sg['mesh']['thick']
    G = np.zeros((2, 2))
    for k in range(t.size):
        G += ks * t[k] * np.array([[C[k, 4, 4], C[k, 4, 3]],
                                   [C[k, 3, 4], C[k, 3, 3]]])
    return G


def fsdt_mr(sg, p, q, q11, n_out=41, ks=5.0 / 6.0):
    """The analytic FSDT + Mendonca-Ruviaro recovery chain.  All amplitudes."""
    d, Ehat, ghat = navier_solve(sg['A6'], fsdt_shear2(sg, ks), p, q, q11)
    U0, V0, W0, Px, Py = d
    H = float(sg['mesh']['thick'].sum())

    z, e, _ = SG.sample_points(sg['mesh'], n_out)
    lay = sg['mesh']['elem_layer'][e]
    C = np.asarray(sg['C_layers'])
    Slay = np.stack([np.linalg.inv(C[k]) for k in range(C.shape[0])])
    Qb = np.stack([np.asarray(plane_stress_C(jnp.asarray(C[k])))
                   for k in range(C.shape[0])])
    assert np.max(np.abs(Qb[:, 0, 2])) < 1e-3 * np.max(np.abs(Qb)), \
        "recovery formulas assume globally-orthotropic layers"

    # ---- FSDT in-plane stress amplitudes at each z (their Eq. 18) ----
    e0 = Ehat[:3]; ka = Ehat[3:]
    sx = np.empty_like(z); sy = np.empty_like(z); txy = np.empty_like(z)
    ez_row = np.empty((z.size, 3))               # [S13, S23, S33] per point
    g13c = np.empty_like(z)                      # global gamma13 compliance
    for i, (zz, k) in enumerate(zip(z, lay)):
        eps3 = np.array([e0[0] + zz * ka[0], e0[1] + zz * ka[1], e0[2] + zz * ka[2]])
        s3 = Qb[k] @ eps3
        sx[i], sy[i], txy[i] = s3[0], s3[1], s3[2]
        ez_row[i] = Slay[k][2, [0, 1, 2]]
        g13c[i] = Slay[k][4, 4]

    # ---- transverse shear by equilibrium integration (their Eqs. 14-15) ----
    txz = _cumtrapz(-p * sx + q * txy, z)        # CS amplitude at A
    tyz = _cumtrapz(p * txy - q * sy, z)         # SC amplitude at C
    # linear correction, their Eq. 23 (vanishes identically for the analytic solution)
    txz = txz - txz[-1] * (0.5 + z / H)
    tyz = tyz - tyz[-1] * (0.5 + z / H)

    # ---- sigma_z by equilibrium + multiplicative scaling (their Eqs. 24, 32) ----
    sz = _cumtrapz(p * txz + q * tyz, z)
    sz_top = sz[-1]
    sz = sz * (q11 / sz_top)

    # ---- w: eps_z from the 3-D compliance, integrate, shift (their Eqs. 33-35) ----
    epsz = ez_row[:, 0] * sx + ez_row[:, 1] * sy + ez_row[:, 2] * sz
    wi = _cumtrapz(epsz, z)
    i0 = int(np.argmin(np.abs(z)))               # reference surface z = 0
    w = wi - wi[i0] + W0

    # ---- u: gamma_xz from the 3-D compliance, integrate, shift+rotate (36-44) ----
    gxz = g13c * txz
    wx = p * w                                   # w_ic,x amplitude (CS family)
    ui = _cumtrapz(gxz - wx, z)
    uis = ui - ui[i0] + U0
    ax = Px - (12.0 / H ** 3) * np.trapezoid(z * uis, z)
    u = uis + z * ax

    return dict(z=z, d=d, txz=txz, tyz=tyz, sz=sz, u=u, w=w, sx=sx, sy=sy,
                sz_top_raw=sz_top / q11)


# ----------------------------------------------------------------------- driver
def run_case(case, aspect, q11=1.0e4, a=1.0, npl_sg=6, order=3, n_out=41):
    """Solve exact + both models for one laminate at a/H = ``aspect``."""
    H = a / aspect
    thick, ang, mats = CASES[case](H)
    ex = ExactNavier(thick, ang, mats, MATDB_MR, a, a, q11=q11)
    sg = SG.build(thick, ang, mats, MATDB_MR, n_per_layer=npl_sg, elem_order=order)
    p = np.pi / a; q = np.pi / a

    zc, sig_e, eps_e, uvw_e = ex.profile(n_per_layer=n_out)
    ms = msg_mr(sg, p, q, q11, n_out=n_out)
    fs = fsdt_mr(sg, p, q, q11, n_out=n_out)
    assert np.allclose(zc, ms['z'], atol=1e-6 * H)

    nrm = dict(t=H / (q11 * a), s=1.0 / q11,
               u=H ** 2 * E2_REF / (q11 * a ** 3),
               w=100.0 * H ** 3 * E2_REF / (q11 * a ** 4))
    return dict(case=case, aspect=aspect, H=H, a=a, zc=zc, zbar=zc / H, nrm=nrm,
                exact=dict(txz=sig_e[:, 4], tyz=sig_e[:, 3], sz=sig_e[:, 2],
                           u=uvw_e[:, 0], w=uvw_e[:, 2], sx=sig_e[:, 0]),
                msg=ms, fsdt=fs, ex=ex, sg=sg)


def relerr(a_, b_):
    a_ = np.asarray(a_, float); b_ = np.asarray(b_, float)
    return float(np.linalg.norm(a_ - b_) / (np.linalg.norm(b_) + 1e-300))
