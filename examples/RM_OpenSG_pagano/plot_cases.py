"""plot_cases.py -- the archive's plot generator: for each of the four curated
Pagano cases AND each of two thickness ratios (L/h = 4 and L/h = 10), compare
THREE methods through the thickness --

    OpenSG-RM     the full recovery chain (stress + displacement: load
                  ladders, equilibrium sigma_33, mean-zero warping +
                  Kirchhoff displacement composition)
    Whitney-1973  first-order shear deformation theory with the Whitney shear
                  correction: classical-lamination in-plane staircase, the
                  constitutive transverse-shear staircase, sigma_33 = 0, and
                  plate-kinematics displacements (u + z phi, w constant)
    Pagano        the exact 3-D elasticity solution of the same problem

-- and save EVERY component as its OWN figure into per-thickness subfolders:

    <case>/L_h_4/plot_s11.png ... plot_U3.png  +  three_method.dat
    <case>/L_h_10/...                             (same set, thinner plate)

Identically-zero fields (cross-ply sigma_12 / sigma_23 / U2) are plotted at
their ACTUAL values -- the axis multiplier (x 1e-14 style) plus an in-figure
"max |...| = ... (numerical zero)" annotation make the magnitude explicit.

Run:  python examples/RM_OpenSG_pagano/plot_cases.py  [--case NAME]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "garg"))
sys.path.insert(0, os.path.join(ROOT, "examples", "pagano_recovery"))

from opensg_jax.fe_jax.msg_rm_plate import (rm_plate_msg,             # noqa: E402
                                            msgrm_strain_at_depth,
                                            msgrm_warping_at_depth)
from opensg_jax.fe_jax.msg_materials import rotated_stiffness_6x6     # noqa: E402
from pagano_exact import ExactCyl                                     # noqa: E402
from recovery_bench import (plate_dofs_theory, harmonic_ops,          # noqa: E402
                            clt_blocks, fsdt_inplane, _grid)

GARG_DB = {"pagano": {"E": [25.0e9, 1.0e9, 1.0e9], "G": [0.5e9, 0.5e9, 0.2e9],
                      "nu": [0.25, 0.25, 0.25], "rho": 1.0},
           "face": {"E": [131.0e9, 10.34e9, 10.34e9],
                    "G": [6.205e9, 6.205e9, 3.0e9],
                    "nu": [0.22, 0.22, 0.22], "rho": 1.0},
           "core": {"E": [0.5776e9] * 3, "G": [0.1079e9] * 3,
                    "nu": [0.0025] * 3, "rho": 1.0}}
YU_DB = {"yu": {"E": [25.0e6, 1.0e6, 1.0e6], "G": [0.5e6, 0.5e6, 0.2e6],
                "nu": [0.25, 0.25, 0.25], "rho": 1.0}}

# fr = ply thickness FRACTIONS of h (bottom first); qt/qb = face-pressure
# fractions of q0 (garg = top-loaded, yu = the split face load)
CASES_LOCAL = {
    "garg_caseA": dict(fr=(1 / 3,) * 3, ang=(0.0, 90.0, 0.0),
                       mats=("pagano",) * 3, db=GARG_DB, a=1.0, q0=1.0e4,
                       qt=1.0, qb=0.0, unit=" [Pa]", ulab="m",
                       label="garg caseA [0/90/0]"),
    "garg_caseC": dict(fr=(0.1, 0.8, 0.1), ang=(0.0, 0.0, 0.0),
                       mats=("face", "core", "face"), db=GARG_DB, a=1.0,
                       q0=1.0e4, qt=1.0, qb=0.0, unit=" [Pa]", ulab="m",
                       label="garg caseC [0/core/0] sandwich"),
    "yu2003_case1": dict(fr=(0.5, 0.5), ang=(15.0, -15.0), mats=("yu",) * 2,
                         db=YU_DB, a=4.0, q0=1.0, qt=0.5, qb=-0.5,
                         unit="$/p_0$", ulab="in",
                         label="yu2003 case1 [15/-15]"),
    "yu2003_case2": dict(fr=(0.25,) * 4, ang=(30.0, -30.0, -30.0, 30.0),
                         mats=("yu",) * 4, db=YU_DB, a=4.0, q0=1.0, qt=0.5,
                         qb=-0.5, unit="$/p_0$", ulab="in",
                         label="yu2003 case2 [30/-30/-30/30]"),
}
S_LIST = (4.0, 10.0)


def build(name, S):
    """Assemble one (case, L/h) configuration: laminate scaled to h = a/S,
    the exact solver with the case's face loads, and the OpenSG-RM law."""
    c = dict(CASES_LOCAL[name])
    a = c["a"]; h = a / S
    thk = [f * h for f in c["fr"]]
    ang = list(c["ang"]); mats = list(c["mats"]); db = c["db"]
    ex = ExactCyl(thk, ang, mats, db, a, q0=c["qt"] * c["q0"],
                  q_bot=c["qb"] * c["q0"])
    r = rm_plate_msg(thk, ang, mats, db, fraction=0.5)
    ABDG = np.asarray(r["ABDG"])
    c.update(name=name, S=S, h=h, thk=thk, ang=ang, mats=mats, p=np.pi / a,
             ex=ex, r=r, A6=np.asarray(r["A6"]), G2=ABDG[6:8, 6:8])
    return c


def fsdt_chain(cf, zc):
    """The standalone Whitney-1973 FSDT solution at both stations (classical
    ABD + Whitney-corrected shear, its own harmonic plate solve, CLT in-plane
    staircase, constitutive shear staircase, plate-kinematics displacements)."""
    thk, ang, mats, db = cf["thk"], cf["ang"], cf["mats"], cf["db"]
    p, q0 = cf["p"], cf["q0"]
    A6c, G2w, Qlist, zpl, k1sq = clt_blocks(thk, ang, mats, db)
    y, Es = plate_dofs_theory(A6c, G2w, p, q0)
    _, Bg = harmonic_ops(p)
    Q12 = G2w @ (Bg @ y)
    s_in = fsdt_inplane(zc, Qlist, zpl, Es)
    ply = np.clip(np.searchsorted(zpl[1:-1], zc, side="left"), 0,
                  len(thk) - 1)
    Csh = []
    for m, x in zip(mats, ang):
        C = np.asarray(rotated_stiffness_6x6(db[m]["E"], db[m]["G"],
                                             db[m]["nu"], x))
        Csh.append(np.array([[C[4, 4], C[3, 4]], [C[3, 4], C[3, 3]]]))
    gam_avg = np.linalg.solve(G2w, Q12)
    s_sh = np.array([Csh[k] @ gam_avg for k in ply])
    Ufs = np.column_stack([y[0] + zc * y[3], y[1] + zc * y[4],
                           np.full_like(zc, y[2])])
    return dict(s_in=s_in, s_sh=s_sh, U=Ufs, k1sq=k1sq)


def rm_chain(cf, zc):
    """The OpenSG-RM chain (identical composition to the tutorials)."""
    p, q0 = cf["p"], cf["q0"]
    y, Es = plate_dofs_theory(cf["A6"], cf["G2"], p, q0)
    z6 = np.zeros(6)
    dE1, dE11 = p * Es, -p ** 2 * Es
    qt_e = np.array([0, p, 0, 0, 0, 0.0]) * (cf["qt"] * q0)
    qb_e = np.array([0, p, 0, 0, 0, 0.0]) * (cf["qb"] * q0)
    qt_m = np.array([1, 0, 0, -p ** 2, 0, 0.0]) * (cf["qt"] * q0)
    qb_m = np.array([1, 0, 0, -p ** 2, 0, 0.0]) * (cf["qb"] * q0)
    n = len(zc)
    S_end = np.empty((n, 6)); S_mid = np.empty((n, 6)); U = np.empty((n, 3))
    for i, z in enumerate(zc):
        S_end[i] = msgrm_strain_at_depth(cf["r"], z, z6, dE1,
                                         qt6=qt_e, qb6=qb_e)[1]
        S_mid[i] = msgrm_strain_at_depth(cf["r"], z, Es, dE11=dE11,
                                         qt6=qt_m, qb6=qb_m)[1]
        w0 = msgrm_warping_at_depth(cf["r"], z, z6, dE1, qt6=qt_e, qb6=qb_e)
        wm = msgrm_warping_at_depth(cf["r"], z, Es, dE11=dE11,
                                    qt6=qt_m, qb6=qb_m)
        U[i] = [y[0] - z * p * y[2] + w0[0], y[1] + w0[1], y[2] + wm[2]]
    s33 = cf["qb"] * q0 + np.concatenate(
        [[0.0], np.cumsum(0.5 * p * (S_end[1:, 4] + S_end[:-1, 4])
                          * np.diff(zc))])
    return dict(S_end=S_end, S_mid=S_mid, s33=s33, U=U)


def run_case(name, S):
    """One (case, L/h): compute the three chains, write the nine individual
    figures and three_method.dat into <case>/L_h_<S>/."""
    cf = build(name, S)
    h = cf["h"]
    zc = _grid(cf["thk"])
    rm = rm_chain(cf, zc)
    fs = fsdt_chain(cf, zc)
    ze, sige, _, uvwe = cf["ex"].profile(n_per_layer=81)
    exs = np.column_stack([np.interp(zc, ze, sige[:, j]) for j in range(6)])
    exu = np.column_stack([np.interp(zc, ze, uvwe[:, j]) for j in range(3)])

    su = cf["unit"]; ul = " [%s]" % cf["ulab"]
    panels = {
        "s11": (rm["S_mid"][:, 0], fs["s_in"][:, 0], exs[:, 0],
                r"$\sigma_{11}$%s  at  $x=a/2$" % su),
        "s22": (rm["S_mid"][:, 1], fs["s_in"][:, 1], exs[:, 1],
                r"$\sigma_{22}$%s  at  $x=a/2$" % su),
        "s12": (rm["S_mid"][:, 5], fs["s_in"][:, 2], exs[:, 5],
                r"$\sigma_{12}$%s  at  $x=a/2$" % su),
        "s13": (rm["S_end"][:, 4], fs["s_sh"][:, 0], exs[:, 4],
                r"$\sigma_{13}$%s  at  $x=0$" % su),
        "s23": (rm["S_end"][:, 3], fs["s_sh"][:, 1], exs[:, 3],
                r"$\sigma_{23}$%s  at  $x=0$" % su),
        "s33": (rm["s33"], np.zeros_like(zc), exs[:, 2],
                r"$\sigma_{33}$%s  at  $x=a/2$" % su),
        "U1": (rm["U"][:, 0], fs["U"][:, 0], exu[:, 0],
               r"$U_1$%s  at  $x=0$" % ul),
        "U2": (rm["U"][:, 1], fs["U"][:, 1], exu[:, 1],
               r"$U_2$%s  at  $x=0$" % ul),
        "U3": (rm["U"][:, 2], fs["U"][:, 2], exu[:, 2],
               r"$U_3$%s  at  $x=a/2$" % ul),
    }
    # family scales, used only to RECOGNIZE numerically-zero fields; those are
    # plotted at their ACTUAL values with the magnitude annotated in-figure
    fam_scale = {"s11": np.max(np.abs(exs[:, 0])),
                 "s22": np.max(np.abs(exs[:, 0])),
                 "s12": np.max(np.abs(exs[:, 0])),
                 "s13": np.max(np.abs(exs[:, 4])),
                 "s23": np.max(np.abs(exs[:, 4])),
                 "s33": np.max(np.abs(exs[:, 2])),
                 "U1": np.max(np.abs(exu)), "U2": np.max(np.abs(exu)),
                 "U3": np.max(np.abs(exu))}
    outdir = os.path.join(HERE, name, "L_h_%g" % S)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for key, (m_rm, m_fs, m_ex, xlabel) in panels.items():
        scale = fam_scale[key]
        vmax = max(np.max(np.abs(m_ex)), np.max(np.abs(m_rm)))
        zero_field = vmax < 1e-6 * scale
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        ax.plot(m_ex, zc / h, "-", color="k", lw=2.0, label="Pagano exact 3-D")
        ax.plot(m_rm, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none",
                mew=1.2, lw=1.6, markevery=10, label="OpenSG-RM")
        ax.plot(m_fs, zc / h, "--", color="#1f77b4", lw=1.5,
                label="Whitney-1973")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
        if zero_field:
            ax.text(0.5, 0.03,
                    "max |value| = %.1e  (numerical zero: ~1e%d of the "
                    "dominant field)"
                    % (vmax,
                       int(np.floor(np.log10(max(vmax / scale, 1e-300))))),
                    transform=ax.transAxes, ha="center", fontsize=8.5,
                    color="0.35")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("$z/h$", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "plot_%s.png" % key), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    hdr = ["%s, L/h = %g -- three-method comparison (OpenSG-RM / "
           "Whitney-1973 FSDT / Pagano exact 3-D)" % (cf["label"], S),
           "stations: s11 s22 s12 s33 U3 at x = a/2; s13 s23 U1 U2 at x = 0",
           "Whitney k1^2 = %.6f (used inside the FSDT shear staircase)"
           % fs["k1sq"],
           "columns: z[%s] then (RM, FSDT, exact) triplets for "
           "s11 s22 s12 s13 s23 s33 U1 U2 U3" % cf["ulab"]]
    cols = [zc]
    for key in ("s11", "s22", "s12", "s13", "s23", "s33", "U1", "U2", "U3"):
        m_rm, m_fs, m_ex, _ = panels[key]
        cols += [m_rm, m_fs, m_ex]
    np.savetxt(os.path.join(outdir, "three_method.dat"),
               np.column_stack(cols), header="\n".join(hdr), fmt="%15.6e")
    print("  %-14s L/h = %-3g -> %s/L_h_%g/plot_*.png + three_method.dat"
          % (name, S, name, S))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="one case (default: all)")
    args = ap.parse_args()
    print("three-method comparison plots, L/h = 4 and 10, individual files")
    for nm in ([args.case] if args.case else list(CASES_LOCAL)):
        for S in S_LIST:
            run_case(nm, S)
