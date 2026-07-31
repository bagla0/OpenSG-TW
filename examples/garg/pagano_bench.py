"""pagano_bench.py -- the orchestrator for the RM-vs-Pagano benchmarks (driven by
examples/garg/case*/7_helper_RM_Pagano_benchmark{1,2,3}.py).

Exactly three standalone chains, one module each:

  pagano_exact.py   the exact 3-D reference (Pagano 1969 / Garg Eqs. 18-24) --
                    REFERENCE CURVES ONLY, never an input to the other chains
  statics_fsdt.py   the statics of Q1/M11 (full derivation in its docstring) and the
                    standalone FSDT chain with the Whitney-1973 Eq.-(7) k1^2
  the MSG core      rm_plate_msg (8x8 at the mid-surface) + msgrm_strain_at_depth
                    (Eq.-63 recovery), driven by the SAME statics Q1

Stations (the harmonic families): sigma_13 at x = 0 (cos peak, Q1 = q0/p there);
sigma_33 at x = a/2 (sin peak; from thickness equilibrium of the recovered sigma_13
amplitude).  FSDT has no sigma_33 (plane-stress plies).  No Abaqus content anywhere:
a shell FE cannot predict a through-thickness stress standalone; where statics cannot
supply the resultants, an Abaqus run may provide FF -- carrying the SAME section law
as the recovery chain it feeds.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CC)
sys.path.insert(0, HERE)

from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth  # noqa: E402
from opensg_jax.fe_jax.msg_transverse_shear import transverse_shear_stiffness   # noqa: E402
from pagano_exact import pagano_profiles                              # noqa: E402
from statics_fsdt import statics_resultants, fsdt_s13                 # noqa: E402
from garg_layups import MATERIAL_DB, LAYUPS, H                        # noqa: E402

q0 = 1.0e4
a = 1.0
p = np.pi / a


def run_case(case, S, tag):
    """One (layup, S) benchmark.  Writes pagano_S<S>.dat + pagano_S<S>.png into the
    case folder and returns the error summary dict."""
    outdir = os.path.join(HERE, case)
    h = a / S
    lay = LAYUPS[case]
    fr = [t / H for t in lay["thick"]]
    thk = [f * h for f in fr]; ang = lay["angles"]; mats = lay["mat_names"]

    # chain 1: the exact reference (curves only)
    zc, sig, uvw = pagano_profiles(thk, ang, mats, MATERIAL_DB, a=a, q0=q0)
    w_ex = float(uvw[np.argmin(np.abs(zc)), 2])

    # the shared input: statics of the problem statement (statics_fsdt sec. 1)
    Q1, _ = statics_resultants(q0, a, 0.0)            # x = 0: Q1 = q0/p, M11 = 0
    FF_end = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Q1, 0.0])

    # chain 2: MSG-RM (8x8 inversion -> gradient closure -> Eq.-66 recovery WITH the
    # load ladder: sigma33 comes DIRECTLY from the constitutive law, Yu's route)
    r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
    D11 = float(r["ABDG"][3, 3]); G11 = float(r["ABDG"][6, 6])
    w_msg = q0 / (p ** 4 * D11) + q0 / (p ** 2 * G11)
    S6 = np.linalg.inv(np.asarray(r["A6"]))
    # sigma13 station x = 0:  E(0) = 0, E,1 = S6 [0,0,0,Q1,0,0]; load: q(0) = 0,
    # q,1(0) = p q0 (cos family) -> the V1L/V2L1 load terms are active
    E6_end = S6 @ FF_end[:6]
    dE1_end = S6 @ np.array([0, 0, 0, Q1, 0, 0.0])
    # sigma33 station x = a/2:  E = S6 [0,0,0,M11,0,0], E,11 = -p^2 E (sin family);
    # load: q = q0, q,11 = -p^2 q0.  Direct recovery -- no equilibrium integration.
    _, M11_mid = statics_resultants(q0, a, a / 2.0)
    E6_mid = S6 @ np.array([0, 0, 0, M11_mid, 0, 0.0])
    dE11_mid = -p * p * E6_mid
    z6 = np.zeros(6)
    s13_m = np.empty_like(zc); s33_m = np.empty_like(zc)
    for i, z in enumerate(zc):
        s13_m[i] = msgrm_strain_at_depth(r, z, E6_end, dE1_end, z6,
                                         dq1=p * q0)[1][4]
        s33_m[i] = msgrm_strain_at_depth(r, z, E6_mid, None, None, dE11=dE11_mid,
                                         q=q0, dq11=-p * p * q0)[1][2]
    # cross-check column: the thickness-equilibrium route (leading-order equivalent)
    s33_eq = np.concatenate([[0.0], np.cumsum(0.5 * p * (s13_m[1:] + s13_m[:-1])
                                              * np.diff(zc))])

    # chain 3: standalone FSDT with the Whitney-1973 k (statics_fsdt secs. 2-3)
    s13_f, k1sq, A55 = fsdt_s13(zc, thk, ang, mats, MATERIAL_DB, Q1)
    G_w = np.asarray(transverse_shear_stiffness(thk, ang, mats, MATERIAL_DB)[0])

    def relerr(m, e):
        return 100 * np.linalg.norm(m - e) / np.linalg.norm(e)

    e13 = relerr(s13_m, sig[:, 4]); e13f = relerr(s13_f, sig[:, 4])
    e33 = relerr(s33_m, sig[:, 2]); e33eq = relerr(s33_eq, sig[:, 2])

    # ------------------------------------------------------------------- .dat
    hdr = ["%s -- MSG-RM vs EXACT (Pagano) benchmark %s,  S = a/h = %g" % (case, tag, S),
           "reference surface: MID-SURFACE (fraction = 0.5) everywhere",
           "stations: x = 0 (s13, Q1 = q0/p), x = a/2 (s33)",
           "plies: " + ", ".join("%s(%.4gmm/%g)" % (m, 1e3 * t, x)
                                 for m, t, x in zip(mats, thk, ang)),
           "",
           "RM 8x8 ABDG (rows e11,e22,g12,k11,k22,k12,2g13,2g23):"]
    hdr += ["  " + " ".join("%14.6e" % v for v in row) for row in np.asarray(r["ABDG"])]
    hdr += ["",
            "INPUT FROM STATICS ONLY (statics_fsdt.py sec. 1): Q1(0) = q0/p = %.6g" % Q1,
            "  FF_end (x=0) = [%s]" % ", ".join("%.6g" % v for v in FF_end),
            "  plate w(a/2) = %.6e ; exact w = %.6e" % (w_msg, w_ex),
            "",
            "transverse-shear stiffness, the two constructions:",
            "  G_whitney (complementary energy, Whitney JAM 40 (1973) Eq. (7)):",
            "    [[%13.6e, %13.6e], [%13.6e, %13.6e]]"
            % (G_w[0, 0], G_w[0, 1], G_w[1, 0], G_w[1, 1]),
            "  G_msg (MSG-RM least-squares projection, Yu 2003 Eq. 61):",
            "    [[%13.6e, %13.6e], [%13.6e, %13.6e]]"
            % (r["G_msg"][0, 0], r["G_msg"][0, 1], r["G_msg"][1, 0], r["G_msg"][1, 1]),
            "  k1^2 (statics_fsdt.whitney_k1sq, used in the FSDT curve): %.6f" % k1sq,
            "  (uniform-k reference 5/6 = 0.8333;  A55 = %.6e)" % A55,
            "",
            "THREE STANDALONE CHAINS, out-of-plane stresses only:",
            "  exact   Pagano cyl. bending (JCM 3 (1969) 398-411; Garg Eqs. 18-24)",
            "  MSG-RM  statics -> 8x8 inversion -> Eq.-66 recovery WITH the load",
            "          ladder (V1L/V2L, Yu Eqs. 29/45): s33 DIRECT from the",
            "          constitutive law -- Yu's route, no equilibrium integration",
            "  FSDT    statics Q1 -> staircase C55(z) Q1/(k1^2 A55); NO s33 in FSDT",
            "rel L2 errors vs exact:  s13 %7.3f%%  (FSDT-Whitney %7.2f%%)"
            % (e13, e13f),
            "  s33 DIRECT %7.3f%%   (equilibrium-integration cross-check %7.3f%%)"
            % (e33, e33eq),
            "s33 faces (bot, top)/q0 = (%+.4f, %+.4f)   (direct recovery)"
            % (s33_m[0] / q0, s33_m[-1] / q0),
            "",
            "columns: z[m]  s13_msg  s13_exact  s13_fsdt  s33_msg_direct  s33_msg_equil"
            "  s33_exact  [Pa]"]
    np.savetxt(os.path.join(outdir, "pagano_S%g.dat" % S),
               np.column_stack([zc, s13_m, sig[:, 4], s13_f, s33_m, s33_eq, sig[:, 2]]),
               header="\n".join(hdr), fmt="%15.6e")

    # ------------------------------------------------------------------- plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5.2))
    ax1.plot(sig[:, 4], zc / h, "-", color="k", lw=2.0, label="exact 3-D (Pagano)")
    ax1.plot(s13_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4, label="MSG-RM")
    ax1.plot(s13_f, zc / h, "--", color="#1f77b4", lw=1.4,
             label="FSDT constitutive\n(Whitney-1973 $k_1^2$)")
    ax1.set_xlabel(r"$\sigma_{13}$ [Pa]  at  $x=0$", fontsize=11)
    ax1.set_ylabel("$z/h$", fontsize=11)
    ax2.plot(sig[:, 2], zc / h, "-", color="k", lw=2.0)
    ax2.plot(s33_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4)
    ax2.set_xlabel(r"$\sigma_{33}$ [Pa]  at  $x=a/2$", fontsize=11)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    # vertical legend OUTSIDE the axes (right of the second panel): it can never
    # block a curve, whatever shape the profiles take across the nine cases
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pagano_S%g.png" % S), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dict(case=case, S=S, e13=e13, e13f=e13f, e33=e33, e33eq=e33eq,
                ABDG=np.asarray(r["ABDG"]))


def run_benchmark(case, S_list, tag):
    print("%s (benchmark %s): stations x=0 (s13) and x=a/2 (s33), mid-surface reference"
          % (case, tag))
    out8 = ["RM 8x8 ABDG for %s (mid-surface reference; rows e11,e22,g12,k11,k22,"
            "k12,2g13,2g23)" % case]
    for S in S_list:
        m = run_case(case, S, tag)
        print("  S = %-4g  s13 %7.3f%% (FSDT %7.2f%%)   s33 DIRECT %7.3f%% (equil %7.3f%%)"
              % (S, m["e13"], m["e13f"], m["e33"], m["e33eq"]))
        out8.append("")
        out8.append("S = a/h = %g  (h = %g m):" % (S, a / S))
        out8 += ["  " + " ".join("%14.6e" % v for v in row) for row in m["ABDG"]]
    with open(os.path.join(HERE, case, "rm_8x8.out"), "w") as f:
        f.write("\n".join(out8) + "\n")
