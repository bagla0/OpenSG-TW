"""plot_cases.py -- the archive's plot generator: for each of the four curated
Pagano cases, compare THREE methods through the thickness --

    OpenSG-RM   the full recovery chain (stress + displacement, the same
                composition as the tutorials: load ladders, equilibrium
                sigma_33, mean-zero warping + Kirchhoff displacement)
    FSDT        first-order shear deformation theory with the Whitney shear
                correction: classical lamination in-plane stress, the
                constitutive transverse-shear staircase, sigma_33 = 0, and
                plate-kinematics displacements (u + z phi, w constant)
    Pagano      the exact 3-D elasticity solution of the same problem

-- and save EVERY component as its OWN figure:

    <case>/plot_s11.png ... plot_s12.png       in-plane stresses (x = a/2)
    <case>/plot_s13.png, plot_s23.png          transverse shear (x = 0)
    <case>/plot_s33.png                        transverse normal (x = a/2)
    <case>/plot_U1.png, plot_U2.png, plot_U3.png   displacements
    <case>/three_method.dat                    all curves as columns

Run:  python examples/RM_OpenSG_pagano/plot_cases.py  [--case NAME]

Functions
---------
fsdt_chain(cf)   the standalone FSDT solution of the case's strip: classical
                 ABD + Whitney-corrected shear stiffness, its own 5-DOF
                 harmonic plate solve, then pointwise in-plane (CLT
                 staircase), transverse-shear (constitutive staircase from
                 [Q1, Q2]), sigma_33 = 0, and plate-kinematics displacements
rm_chain(cf)     the OpenSG-RM chain of the case (identical to full_field)
run_case(name)   compute the three chains + write the 9 figures and the .dat
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
sys.path.insert(0, os.path.join(ROOT, "examples", "yu2003"))
sys.path.insert(0, os.path.join(ROOT, "examples", "pagano_recovery"))

from opensg_jax.fe_jax.msg_rm_plate import (msgrm_strain_at_depth,    # noqa: E402
                                            msgrm_warping_at_depth)
from opensg_jax.fe_jax.msg_materials import rotated_stiffness_6x6     # noqa: E402
from recovery_bench import (CASES, case_setup, plate_dofs_theory,     # noqa: E402
                            harmonic_ops, clt_blocks, fsdt_inplane, _grid)
from statics_fsdt import whitney_k1sq                                 # noqa: E402

OUTDIRS = {"garg_caseA": "garg_caseA", "garg_caseC": "garg_caseC",
           "yu2003_case1": "yu2003_case1", "yu2003_case2": "yu2003_case2"}


def fsdt_chain(cf, zc):
    """The standalone FSDT solution of the case's strip at both stations.

    Variables
    ---------
    A6c, G2w    classical lamination 6x6 and the Whitney-corrected shear 2x2
                (k1^2 x the ply-integrated shear block, from clt_blocks)
    y, Es       the FSDT harmonic plate solve of the strip (same 5-DOF solve,
                FSDT section law) -- supplies [Q1, Q2] and the plate strains
    Q12         the transverse-shear resultants of the FSDT solution
    gam         the (constant) transverse shear strains G2w^-1 [Q1, Q2]
    s_in        (n, 3) in-plane CLT staircase Qbar(z) (e0 + z k)
    s_sh        (n, 2) constitutive staircase [[C55, C45], [C45, C44]](z) gam
    Ufs         (n, 3) FSDT plate-kinematics displacements: U1 = u + z phi1,
                U2 = v + z phi2 (cos family at x = 0), U3 = w (sin, x = a/2)
    """
    thk, ang, mats, db = cf["thk"], cf["ang"], cf["mats"], cf["db"]
    p, q0 = cf["p"], cf["q0"]
    A6c, G2w, Qlist, zpl, k1sq = clt_blocks(thk, ang, mats, db)
    y, Es = plate_dofs_theory(A6c, G2w, p, q0)
    _, Bg = harmonic_ops(p)
    Q12 = G2w @ (Bg @ y)
    gam = np.linalg.solve(G2w, Q12) * k1sq / k1sq          # = Bg @ y
    s_in = fsdt_inplane(zc, Qlist, zpl, Es)
    ply = np.clip(np.searchsorted(zpl[1:-1], zc, side="left"), 0,
                  len(thk) - 1)
    Csh = []
    for m, x in zip(mats, ang):
        C = np.asarray(rotated_stiffness_6x6(db[m]["E"], db[m]["G"],
                                             db[m]["nu"], x))
        Csh.append(np.array([[C[4, 4], C[3, 4]], [C[3, 4], C[3, 3]]]))
    gam_avg = np.linalg.solve(G2w, Q12)                    # k-corrected strain
    s_sh = np.array([Csh[k] @ gam_avg for k in ply])
    Ufs = np.column_stack([y[0] + zc * y[3],               # u + z phi1
                           y[1] + zc * y[4],               # v + z phi2
                           np.full_like(zc, y[2])])        # w (constant in z)
    return dict(s_in=s_in, s_sh=s_sh, U=Ufs, k1sq=k1sq)


def rm_chain(cf, zc):
    """The OpenSG-RM chain (identical composition to the tutorials): stress at
    both stations with the face load ladders, sigma_33 by equilibrium from the
    loaded bottom face, Kirchhoff displacement with mean-zero warping."""
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


def run_case(name):
    """Compute the three chains and write the nine individual figures + .dat."""
    cf = case_setup(name)
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
    # reference scale per panel family, used to RECOGNIZE identically-zero
    # fields (cross-ply sigma_12/sigma_23/U2): without this the axis
    # auto-scales to the ~1e-14-relative numerical noise and machine-zero
    # looks like a wild curve.  Zero fields are drawn on an axis scaled to
    # the family's dominant component so all methods collapse onto zero.
    fam_scale = {"s11": np.max(np.abs(exs[:, 0])),
                 "s22": np.max(np.abs(exs[:, 0])),
                 "s12": np.max(np.abs(exs[:, 0])),
                 "s13": np.max(np.abs(exs[:, 4])),
                 "s23": np.max(np.abs(exs[:, 4])),
                 "s33": np.max(np.abs(exs[:, 2])),
                 "U1": np.max(np.abs(exu)), "U2": np.max(np.abs(exu)),
                 "U3": np.max(np.abs(exu))}
    outdir = os.path.join(HERE, OUTDIRS[name])
    for key, (m_rm, m_fs, m_ex, xlabel) in panels.items():
        scale = fam_scale[key]
        zero_field = (max(np.max(np.abs(m_ex)), np.max(np.abs(m_rm)))
                      < 1e-6 * scale)
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        ax.plot(m_ex, zc / h, "-", color="k", lw=2.0, label="Pagano exact 3-D")
        ax.plot(m_rm, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none",
                mew=1.2, lw=1.6, markevery=10, label="OpenSG-RM")
        ax.plot(m_fs, zc / h, "--", color="#1f77b4", lw=1.5,
                label="Whitney-1973")
        if zero_field:
            ax.set_xlim(-0.05 * scale, 0.05 * scale)
            ax.text(0.5, 0.06, "identically zero field\n(all methods $=0$ "
                    "at plot scale)", transform=ax.transAxes, ha="center",
                    fontsize=9, color="0.35")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("$z/h$", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "plot_%s.png" % key), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    hdr = ["%s -- three-method comparison (OpenSG-RM / FSDT (Whitney) / "
           "Pagano exact 3-D)" % cf["label"],
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
    print("  %-14s -> %s/plot_{s11..U3}.png + three_method.dat"
          % (name, OUTDIRS[name]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="one case (default: all)")
    args = ap.parse_args()
    print("three-method comparison plots (individual files per component)")
    for nm in ([args.case] if args.case else list(OUTDIRS)):
        run_case(nm)
