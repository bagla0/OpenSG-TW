"""pagano_bench.py -- the shared engine for the RM-vs-Pagano benchmarks
(driven by examples/garg/case*/7_helper_RM_Pagano_benchmark{1,2,3}.py).

Chain per (layup, S): homogenize (8x8 ABDG at the MID-SURFACE, fraction = 0.5)
-> FF from the EXACT solution (ff_from_exact below) -> example-7-style dehom ->
pointwise stress vs the exact 3-D elasticity, plus Garg's FSDT baseline.

ff_from_exact: the "Pagano reaction forces".  NO equilibrium assumption is involved --
the resultants are the DEFINITION integrals of the exact stress amplitudes,
    N = int s dz,  M = int s z dz,  Q = int t dz,
evaluated at a station x via the harmonic families (sin for s11/s22/s33/N/M, cos for
s13/Q).  Equilibrium enters only afterwards, inside the recovery: (a) the internal
gradient closure dE1 = inv(A6) @ [0,0,0,Q1,0,0] (M11,1 = Q1), and (b) sigma33 by
through-thickness integration of the recovered sigma13 amplitude
(d s33_hat/dz = p s13_hat, families s13 ~ cos -> s33 ~ sin).

FSDT baseline (Garg's): same A6, shear stiffness k (5/6) sum(t C55); the constitutive
transverse shear  s13(z) = C55(z) Q1 / (k sum(t C55))  -- layerwise constant, face
tractions violated: exactly the deficiency their GPR (and our warping recovery) fixes.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.dirname(os.path.dirname(HERE))
import sys

sys.path.insert(0, CC)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CC, "examples", "TW-paper", "rm_thickness"))

from exact_cyl import ExactCyl                                        # noqa: E402
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth  # noqa: E402
from opensg_jax.fe_jax.msg_materials import rotated_stiffness_6x6     # noqa: E402
from garg_layups import MATERIAL_DB, LAYUPS, H                        # noqa: E402

q0 = 1.0e4
a = 1.0
p = np.pi / a


def ff_from_exact(zc, sig):
    """The 8x1 resultant vectors at the two Pagano stations, by DIRECT INTEGRATION of
    the exact stress amplitudes (no equilibrium used).  Returns (FF_mid, FF_end):
    FF_mid at x = a/2 (sin peak: N and M live, Q1 = 0), FF_end at x = 0 (cos peak:
    Q1 lives, N = M = 0)."""
    N11 = float(np.trapezoid(sig[:, 0], zc))
    N22 = float(np.trapezoid(sig[:, 1], zc))
    M11 = float(np.trapezoid(sig[:, 0] * zc, zc))
    M22 = float(np.trapezoid(sig[:, 1] * zc, zc))
    Q1 = float(np.trapezoid(sig[:, 4], zc))
    return (np.array([N11, N22, 0.0, M11, M22, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Q1, 0.0]))


def run_case(case, S, tag):
    """One (layup, S) benchmark.  Writes pagano_S<S>.dat + pagano_S<S>.png into the
    case folder and returns the error summary dict."""
    outdir = os.path.join(HERE, case)
    h = a / S
    lay = LAYUPS[case]
    fr = [t / H for t in lay["thick"]]
    thk = [f * h for f in fr]; ang = lay["angles"]; mats = lay["mat_names"]

    # exact solution (amplitudes; z from the MID-surface) + its reaction forces
    ex = ExactCyl(thk, ang, mats, MATERIAL_DB, a, q0=q0)
    zc, sig, _, uvw = ex.profile(n_per_layer=81)
    FF_mid, FF_end = ff_from_exact(zc, sig)
    w_ex = float(uvw[np.argmin(np.abs(zc)), 2])

    # MSG-RM homogenization (mid-surface reference) + the plate's own deflection
    r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
    D11 = float(r["ABDG"][3, 3]); G11 = float(r["ABDG"][6, 6])
    w_msg = q0 / (p ** 4 * D11) + q0 / (p ** 2 * G11)

    # dehom: E6 from FF through the 8x8; gradient closure from Q (equilibrium)
    # OUT-OF-PLANE stresses only (the comparison this benchmark is about): the
    # in-plane sigma11 is a plate-level quantity every theory shares and is omitted.
    S6 = np.linalg.inv(np.asarray(r["A6"]))
    E6_end = S6 @ FF_end[:6]
    dE1_end = S6 @ np.array([0, 0, 0, FF_end[6], 0, 0.0])
    z6 = np.zeros(6)
    s13_m = np.empty_like(zc)
    for i, z in enumerate(zc):
        s13_m[i] = msgrm_strain_at_depth(r, z, E6_end, dE1_end, z6)[1][4]
    s33_m = np.concatenate([[0.0], np.cumsum(0.5 * p * (s13_m[1:] + s13_m[:-1])
                                             * np.diff(zc))])

    # Garg's FSDT baseline, replicated ANALYTICALLY (their sec. 2.1 statement: FSDT
    # "gives a constant value of transverse shear stress across the layer" and "is not
    # able to predict" the transverse normal stress).  No Abaqus data and no
    # equilibrium integration is involved in this curve -- it is the CONSTITUTIVE
    # layerwise-constant s13(z) = C55(z) Q1 / (k sum(t C55)) with k = 5/6.
    C55 = np.array([float(rotated_stiffness_6x6(MATERIAL_DB[m]["E"], MATERIAL_DB[m]["G"],
                                                MATERIAL_DB[m]["nu"], x)[4, 4])
                    for m, x in zip(mats, ang)])
    bot = np.concatenate([[0.0], np.cumsum(thk)]) - 0.5 * h
    ply_of = np.clip(np.searchsorted(bot[1:-1], zc, side="left"), 0, len(thk) - 1)
    G_fsdt = 5.0 / 6.0 * float(np.sum(np.array(thk) * C55))
    s13_f = C55[ply_of] * FF_end[6] / G_fsdt

    def relerr(m, e):
        return 100 * np.linalg.norm(m - e) / np.linalg.norm(e)

    e13 = relerr(s13_m, sig[:, 4])
    e33 = relerr(s33_m, sig[:, 2]); e13f = relerr(s13_f, sig[:, 4])

    # ------------------------------------------------------------------- .dat
    hdr = ["%s -- MSG-RM vs EXACT (Pagano) benchmark %s,  S = a/h = %g" % (case, tag, S),
           "reference surface: MID-SURFACE (fraction = 0.5) in the SG, the plate law",
           "and the exact z origin.  stations: x = a/2 (s11, s33), x = 0 (s13)",
           "plies: " + ", ".join("%s(%.4gmm/%g)" % (m, 1e3 * t, x)
                                 for m, t, x in zip(mats, thk, ang)),
           "",
           "RM 8x8 ABDG (rows e11,e22,g12,k11,k22,k12,2g13,2g23):"]
    hdr += ["  " + " ".join("%14.6e" % v for v in row) for row in np.asarray(r["ABDG"])]
    hdr += ["",
            "FF from the exact solution (DIRECT stress integration, no equilibrium):",
            "  FF_mid (x=a/2) = [%s]" % ", ".join("%.6g" % v for v in FF_mid),
            "  FF_end (x=0)   = [%s]" % ", ".join("%.6g" % v for v in FF_end),
            "  u2d = [0, 0, %.6e] (plate w; exact %.6e)" % (w_msg, w_ex),
            "",
            "OUT-OF-PLANE stresses only.  FSDT s13 = the paper's baseline, replicated",
            "analytically (constitutive layerwise-constant, k = 5/6; FSDT has NO s33).",
            "rel L2 errors vs exact:  s13 %7.3f%%  (FSDT %7.2f%%)   s33 %7.3f%%"
            % (e13, e13f, e33),
            "s33 top-face closure: %.4f q0" % (s33_m[-1] / q0),
            "",
            "columns: z[m]  s13_msg  s13_exact  s13_fsdt  s33_msg  s33_exact  [Pa]"]
    np.savetxt(os.path.join(outdir, "pagano_S%g.dat" % S),
               np.column_stack([zc, s13_m, sig[:, 4], s13_f, s33_m, sig[:, 2]]),
               header="\n".join(hdr), fmt="%15.6e")

    # ------------------------------------------------------------------- plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 5.2))
    ax1.plot(sig[:, 4], zc / h, "-", color="k", lw=2.0, label="exact 3-D (Pagano)")
    ax1.plot(s13_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4, label="MSG-RM")
    ax1.plot(s13_f, zc / h, "--", color="#1f77b4", lw=1.4,
             label="FSDT constitutive (analytic, k=5/6)")
    ax1.set_xlabel(r"$\sigma_{13}$ [Pa]  at  $x=0$", fontsize=11)
    ax1.set_ylabel("$z/h$", fontsize=11)
    ax2.plot(sig[:, 2], zc / h, "-", color="k", lw=2.0)
    ax2.plot(s33_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4)
    ax2.set_xlabel(r"$\sigma_{33}$ [Pa]  at  $x=a/2$", fontsize=11)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=9, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pagano_S%g.png" % S), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dict(case=case, S=S, e13=e13, e13f=e13f, e33=e33,
                ABDG=np.asarray(r["ABDG"]))


def run_benchmark(case, S_list, tag):
    print("%s (benchmark %s): stations x=a/2 and x=0, mid-surface reference" % (case, tag))
    out8 = ["RM 8x8 ABDG for %s (mid-surface reference; rows e11,e22,g12,k11,k22,"
            "k12,2g13,2g23)" % case]
    for S in S_list:
        m = run_case(case, S, tag)
        print("  S = %-4g  s13 %7.3f%% (FSDT %7.2f%%)   s33 %7.3f%%"
              % (S, m["e13"], m["e13f"], m["e33"]))
        out8.append("")
        out8.append("S = a/h = %g  (h = %g m):" % (S, a / S))
        out8 += ["  " + " ".join("%14.6e" % v for v in row) for row in m["ABDG"]]
    with open(os.path.join(HERE, case, "rm_8x8.out"), "w") as f:
        f.write("\n".join(out8) + "\n")
