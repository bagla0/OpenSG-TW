"""run_mendonca_comparison.py -- the Mendonca & Ruviaro (FEAD 260 (2026) 104609)
benchmark through the CORE MSG-RM module, 1D-yaml route end to end.

Their process: FSDT (k = 5/6) FE solution + a sequential SPR-smoothed recovery chain
(equilibrium integration for tau_xz/sigma_z with top-face scaling, compliance-integrated
w and u with shift/rotation corrections, their Eqs. 14-44).  Here the chain is evaluated
ANALYTICALLY (single-harmonic Navier -> the curves their FE + smoothing converges to,
so no plot digitisation is needed), against:

  mendonca_plates.yaml -> msg_mesh.load_yaml -> core rm_plate_msg (homo; 8x8 in the .dat)
  -> Navier plate solution closed with G_MSG -> core msgrm_strain_at_depth /
  msgrm_warping_at_depth (direct recovery, no correction/shift/scaling steps)
  vs the EXACT 3-D elasticity (exact_navier.ExactNavier, Pagano 1970 bidirectional).

Normalisations (their Eq. 52): tbar = H/(q11 a) tau, sbar_z = sigma_z/q11,
ubar = H^2 E2/(q11 a^3) u, wbar = 100 H^3 E2/(q11 a^4) w.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, CC)
sys.path.insert(0, HERE)

from opensg_jax.fe_jax.msg_mesh import load_yaml
from opensg_jax.fe_jax.msg_rm_plate import (rm_plate_msg, msgrm_strain_at_depth,
                                            msgrm_warping_at_depth)
from opensg_jax.fe_jax.msg_transverse_shear import plate_8x8
from exact_navier import ExactNavier

YAML = os.path.join(HERE, "mendonca_plates.yaml")
_, _, MDB, LDB, _ = load_yaml(YAML)

E2_REF = 6.89e9
ASPECTS = (4, 100)
N_OUT = 41
EXACTC, MSGC, MRC = "k", "#ff7f0e", "#1f77b4"
SS = np.array([1., 1., 0., 1., 1., 0.])
CCr = np.array([0., 0., 1., 0., 0., 1.])


# ---------------------------------------------------------------- Navier machinery
def navier_matrices(p, q):
    L = np.zeros((6, 5))
    L[0, 0] = -p; L[1, 1] = -q
    L[2, 0] = q; L[2, 1] = p
    L[3, 3] = -p; L[4, 4] = -q
    L[5, 3] = q; L[5, 4] = p
    Lg = np.zeros((2, 5))
    Lg[0, 2] = p; Lg[0, 3] = 1.0
    Lg[1, 2] = q; Lg[1, 4] = 1.0
    return L, Lg


def navier_solve(A6, G2, p, q, q11):
    L, Lg = navier_matrices(p, q)
    K = L.T @ np.asarray(A6) @ L + Lg.T @ np.asarray(G2) @ Lg
    d = np.linalg.solve(K, np.array([0.0, 0.0, q11, 0.0, 0.0]))
    return d, L @ d, Lg @ d


def point_args(Ehat, p, q):
    """(E6, dE1, dE2) at A=(0,a/2), B=(a/2,a/2), C=(a/2,0)."""
    Ehat = np.asarray(Ehat)
    return ((np.zeros(6), p * SS * Ehat, -q * CCr * Ehat),
            (SS * Ehat, np.zeros(6), np.zeros(6)),
            (np.zeros(6), -p * CCr * Ehat, q * SS * Ehat))


def cumtrapz(y, z):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(z))
    return out


def plane_stress_C(C):
    keep = np.array([0, 1, 5]); drop = np.array([2, 3, 4])
    return (C[np.ix_(keep, keep)]
            - C[np.ix_(keep, drop)] @ np.linalg.solve(C[np.ix_(drop, drop)],
                                                      C[np.ix_(drop, keep)]))


# ---------------------------------------------------------------- the two models
def msg_mr(r, zc, p, q, q11):
    """OpenSG-RM: Navier with G_MSG + direct core recovery (no corrections).
    Stress drivers are the moment-consistent RM measures; the displacement warping is
    taken as the pure through-thickness deviation (the RM director already carries the
    mean shear rotation -- the same rotation gauge as their Eqs. 43-44)."""
    d, Ehat, ghat = navier_solve(r["A6"], r["G_msg"], p, q, q11)
    U0, V0, W0, Px, Py = d
    argA, argB, argC = point_args(Ehat, p, q)
    z = zc
    SigA = np.array([msgrm_strain_at_depth(r, zz, *argA)[1] for zz in z])
    SigB = np.array([msgrm_strain_at_depth(r, zz, *argB)[1] for zz in z])
    SigC = np.array([msgrm_strain_at_depth(r, zz, *argC)[1] for zz in z])
    warpA = np.array([msgrm_warping_at_depth(r, zz, *argA) for zz in z])
    warpB = np.array([msgrm_warping_at_depth(r, zz, *argB) for zz in z])

    txz = SigA[:, 4]; tyz = SigC[:, 3]
    sz = cumtrapz(p * txz + q * tyz, z)

    def dev(w):
        s = np.trapezoid(z * w, z) / np.trapezoid(z * z, z)
        return w - s * z

    u = U0 + z * Px + dev(warpA[:, 0])
    w = W0 + warpB[:, 2]
    return dict(d=d, txz=txz, tyz=tyz, sz=sz, u=u, w=w, sx=SigB[:, 0],
                sz_top_closure=abs(sz[-1] / q11 - 1.0))


def fsdt_mr(r, thk, zc, p, q, q11, ks=5.0 / 6.0):
    """The analytic FSDT + Mendonca-Ruviaro recovery chain (their Eqs. 14-44)."""
    thk = np.asarray(thk, float); H = float(thk.sum())
    C = np.stack([np.asarray(c) for c in r["C_layers"]])
    G2 = np.zeros((2, 2))
    for k in range(thk.size):
        G2 += ks * thk[k] * np.array([[C[k, 4, 4], C[k, 4, 3]],
                                      [C[k, 3, 4], C[k, 3, 3]]])
    d, Ehat, ghat = navier_solve(r["A6"], G2, p, q, q11)
    U0, V0, W0, Px, Py = d

    z = zc
    zb = np.concatenate([[0.0], np.cumsum(thk)]) - 0.5 * H
    lay = np.clip(np.searchsorted(zb[1:-1], z, side="left"), 0, thk.size - 1)
    Slay = np.stack([np.linalg.inv(C[k]) for k in range(C.shape[0])])
    Qb = np.stack([plane_stress_C(C[k]) for k in range(C.shape[0])])

    e0 = Ehat[:3]; ka = Ehat[3:]
    sx = np.empty_like(z); sy = np.empty_like(z); txy = np.empty_like(z)
    ez_row = np.empty((z.size, 3)); g13c = np.empty_like(z)
    for i, (zz, k) in enumerate(zip(z, lay)):
        s3 = Qb[k] @ np.array([e0[0] + zz * ka[0], e0[1] + zz * ka[1], e0[2] + zz * ka[2]])
        sx[i], sy[i], txy[i] = s3
        ez_row[i] = Slay[k][2, [0, 1, 2]]
        g13c[i] = Slay[k][4, 4]

    txz = cumtrapz(-p * sx + q * txy, z)
    tyz = cumtrapz(p * txy - q * sy, z)
    txz = txz - txz[-1] * (0.5 + z / H)              # their Eq. 23
    tyz = tyz - tyz[-1] * (0.5 + z / H)
    sz = cumtrapz(p * txz + q * tyz, z)
    sz = sz * (q11 / sz[-1])                          # their Eq. 32 top-face scaling

    epsz = ez_row[:, 0] * sx + ez_row[:, 1] * sy + ez_row[:, 2] * sz
    wi = cumtrapz(epsz, z)
    i0 = int(np.argmin(np.abs(z)))
    w = wi - wi[i0] + W0                              # their Eq. 35
    gxz = g13c * txz
    ui = cumtrapz(gxz - p * w, z)
    uis = ui - ui[i0] + U0                            # their Eqs. 41-42
    ax = Px - (12.0 / H ** 3) * np.trapezoid(z * uis, z)
    u = uis + z * ax                                  # their Eqs. 43-44 rotation gauge
    return dict(d=d, txz=txz, tyz=tyz, sz=sz, u=u, w=w, sx=sx)


def relerr(a_, b_):
    a_ = np.asarray(a_, float); b_ = np.asarray(b_, float)
    return float(np.linalg.norm(a_ - b_) / (np.linalg.norm(b_) + 1e-300))


# ----------------------------------------------------------------------- driver
rep = ["# Mendonca & Ruviaro (FEAD 260 (2026) 104609) benchmark vs core MSG-RM",
       "# 1D-yaml route: mendonca_plates.yaml -> load_yaml -> core rm_plate_msg + recovery",
       "# exact  = 3-D elasticity (Pagano 1970 bidirectional, double-sine load)",
       "# MR     = the ANALYTIC Mendonca-Ruviaro chain (FSDT k=5/6 + their Eqs. 14-44;",
       "#          the curves their FE + sequential smoothing converges to)",
       "# MSG-RM = Navier closed with G_MSG + direct core warping recovery, no corrections",
       "# quantities: relative L2 profile errors vs exact; their Eq.-52 normalisations",
       "#"]
q11 = 1.0e4; a = 1.0
figdata = {}
for name, lay in LDB.items():
    fr = [float(t) for t in lay["thick"]]
    ang = [float(x) for x in lay["angles"]]
    mats = [str(m) for m in lay["mat_names"]]
    for aspect in ASPECTS:
        H = a / aspect
        thk = [f * H for f in fr]
        r = rm_plate_msg(thk, ang, mats, MDB, fraction=0.5)      # CORE homo
        ex = ExactNavier(thk, ang, mats, MDB, a, a, q11=q11)
        p = np.pi / a; q = np.pi / a
        zc, sig_e, _, uvw_e = ex.profile(n_per_layer=N_OUT)
        ms = msg_mr(r, zc, p, q, q11)
        fs = fsdt_mr(r, thk, zc, p, q, q11)

        nrm = dict(t=H / (q11 * a), s=1.0 / q11,
                   u=H ** 2 * E2_REF / (q11 * a ** 3),
                   w=100.0 * H ** 3 * E2_REF / (q11 * a ** 4))
        exd = dict(txz=sig_e[:, 4], sz=sig_e[:, 2], u=uvw_e[:, 0], w=uvw_e[:, 2])

        rep.append("== %s  a/H = %d  (bc residual %.1e; MSG sz top closure %.1e) =="
                   % (name, aspect, ex.bc_residual(), ms["sz_top_closure"]))
        rep.append("   %-8s %12s %12s" % ("quantity", "MR-chain", "MSG-RM"))
        for key, lab in (("txz", "tbar_xz"), ("sz", "sbar_z"), ("u", "ubar"), ("w", "wbar")):
            rep.append("   %-8s %11.3f%% %11.3f%%"
                       % (lab, 100 * relerr(fs[key], exd[key]), 100 * relerr(ms[key], exd[key])))
        wb = dict(exact=nrm["w"] * exd["w"][np.argmin(np.abs(zc))],
                  mr=nrm["w"] * fs["w"][np.argmin(np.abs(zc))],
                  msg=nrm["w"] * ms["w"][np.argmin(np.abs(zc))])
        rep.append("   central wbar: exact %.4f | MR %.4f | MSG-RM %.4f" %
                   (wb["exact"], wb["mr"], wb["msg"]))
        P8 = plate_8x8(np.asarray(r["A6"]), np.asarray(r["G_msg"]))
        rep.append("   8x8 ABDG (core rm_plate_msg, mid reference; rows e11,e22,g12,"
                   "k11,k22,k12,2g13,2g23):")
        for row in P8:
            rep.append("     " + " ".join("%13.5e" % v for v in row))
        rep.append("")
        print("\n".join(rep[-13 - 8:]), flush=True)
        figdata[(name, aspect)] = (zc / H, exd, ms, fs, nrm)

# reference values read from the MR paper for the [0/90/0] plate (their sec. 5):
rep.append("# MR-paper-read central wbar, [0/90/0]: exact-benchmark 2.0010 (a/H=4), "
           "0.4347 (a/H=100); their FSDT branch 1.774 / 0.4337")

# ---------------------------------------------------------------------- figures
def panel(ax, zbar, xe, xm, xf, lab):
    ax.plot(xe, zbar, "-", color=EXACTC, lw=2.0, label="exact 3-D")
    ax.plot(xf, zbar, "--", color=MRC, lw=1.6, label="Mendonca-Ruviaro chain")
    ax.plot(xm, zbar, ":s", color=MSGC, ms=3, mfc="none", mew=1.1, lw=1.5,
            markevery=6, label="MSG-RM (core)")
    ax.set_xlabel(lab, fontsize=12); ax.set_ylabel(r"$z/H$", fontsize=12)
    ax.grid(alpha=0.3)


for (name, aspect), fname, keys in (
        (("mr_sandwich", 4), "mendonca_sandwich_aH4.png", ("w", "txz", "sz")),
        (("mr_sym", 4), "mendonca_sym_aH4.png", ("u", "txz", "sz"))):
    zbar, exd, ms, fs, nrm = figdata[(name, aspect)]
    labs = dict(w=r"$\bar w$", u=r"$\bar u$", txz=r"$\bar\tau_{xz}$", sz=r"$\bar\sigma_z$")
    scl = dict(w=nrm["w"], u=nrm["u"], txz=nrm["t"], sz=nrm["s"])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, k in zip(axs, keys):
        panel(ax, zbar, scl[k] * np.asarray(exd[k]), scl[k] * np.asarray(ms[k]),
              scl[k] * np.asarray(fs[k]), labs[k])
    axs[0].legend(fontsize=10, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)

open(os.path.join(HERE, "mendonca_results.dat"), "w").write("\n".join(rep) + "\n")
print("wrote mendonca_results.dat + figures", flush=True)
