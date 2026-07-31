"""sig12_sweep.py -- the COMPLETE sigma_12 validation of the OpenSG-RM in-plane
recovery against the exact Pagano solution.

Two claims are established, each with its own evidence:

1. CROSS-PLY (garg caseA [0/90/0], caseC sandwich): the exact cylindrical-
   bending sigma_12 is IDENTICALLY ZERO (specially-orthotropic plies, no
   shear coupling).  The recovered sigma_12 must be zero at machine level --
   reported as the ratio max|s12_rec| / max|s11_rec|, which must sit at the
   1e-14 round-off floor, NOT as a meaningless relative error on a zero field.

2. ANGLE-PLY (Yu material [15/-15] and [30/-30/-30/30]): sigma_12 is a genuine
   layered field.  The recovery error must CONVERGE to zero with the model's
   asymptotic order as the plate gets thinner -- a plateau would mean a real
   defect (a Voigt-row mix-up in the Gamma_l operators, a driver-slot error,
   or a detilt mistake on the g12-coupled columns).  The sweep L/h = 4..64
   measures the observed order for sigma_12 alongside sigma_11.

Run:
    python examples/pagano_recovery/sig12_sweep.py
Writes sig12_sweep.dat + sig12_sweep.png here.

Script variables
----------------
LAYUPS_SWEEP   the two angle-ply layups swept (Yu psi material, equal plies,
               bottom->top) and the two cross-ply spot checks (S = 10)
S_LIST         the aspect ratios a/h of the sweep (span a fixed, h = a/S)
run_point      one (layup, S) in-plane recovery: returns the sigma_11 and
               sigma_12 rel-L2 errors vs exact and the zero-field ratio
orders         observed convergence order between successive S points,
               log(err_i/err_{i+1}) / log(S_{i+1}/S_i) -- ~2 is the expected
               second-order-recovery rate at the sin-peak station
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CC = HERE
while not os.path.isdir(os.path.join(CC, "opensg_jax")):
    _up = os.path.dirname(CC)
    if _up == CC:
        raise RuntimeError("opensg_jax repo root not found above " + __file__)
    CC = _up
sys.path.insert(0, CC)
sys.path.insert(0, os.path.join(CC, "examples", "garg"))
sys.path.insert(0, os.path.join(CC, "examples", "yu2003"))

from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth  # noqa: E402
from pagano_exact import ExactCyl                                     # noqa: E402
import garg_layups                                                    # noqa: E402
import yu_layups                                                      # noqa: E402
from yu_bench import rm_cyl_bend                                      # noqa: E402

A_SPAN = 4.0
Q0 = 1.0
S_LIST = (4.0, 8.0, 16.0, 32.0, 64.0)

LAYUPS_SWEEP = {
    "[15/-15]": dict(fr=(0.5, 0.5), ang=(15.0, -15.0),
                     mats=("yu", "yu"), db=yu_layups.MATERIAL_DB),
    "[30/-30/-30/30]": dict(fr=(0.25,) * 4, ang=(30.0, -30.0, -30.0, 30.0),
                            mats=("yu",) * 4, db=yu_layups.MATERIAL_DB),
}
CROSS_CHECKS = {
    "caseA [0/90/0] S=10": dict(fr=(1 / 3,) * 3, ang=(0.0, 90.0, 0.0),
                                mats=("pagano",) * 3,
                                db=garg_layups.MATERIAL_DB, S=10.0),
    "caseC sandwich S=10": dict(fr=(0.1, 0.8, 0.1), ang=(0.0, 0.0, 0.0),
                                mats=("face", "core", "face"),
                                db=garg_layups.MATERIAL_DB, S=10.0),
}


def run_point(fr, ang, mats, db, S):
    """One (layup, S) in-plane recovery vs exact at x = a/2.

    Variables: fr/ang/mats/db = ply fractions, angles, materials, database;
    S = a/h; thk = fractions scaled to h = a/S; the chain is the standard one
    (rm_plate_msg -> harmonic solve -> Eq. 66 with E6 = Es, E6,11 = -p^2 Es);
    zc = 81-per-ply grid; returns (e11, e12, ratio12) = rel-L2 errors of
    sigma_11 and sigma_12 vs exact plus max|s12_rec|/max|s11_rec| (the
    zero-field diagnostic).
    """
    h = A_SPAN / S
    thk = [f * h for f in fr]
    p = np.pi / A_SPAN
    ex = ExactCyl(list(thk), list(ang), list(mats), db, A_SPAN,
                  q0=0.5 * Q0, q_bot=-0.5 * Q0)
    r = rm_plate_msg(thk, list(ang), list(mats), db, fraction=0.5)
    ABDG = np.asarray(r["ABDG"])
    y, Es, gs, R6, Q = rm_cyl_bend(np.asarray(r["A6"]), ABDG[6:8, 6:8], p, Q0)
    zpl = np.concatenate([[0.0], np.cumsum(thk)]) - h / 2
    zc = np.concatenate([np.linspace(zpl[k] + 1e-12, zpl[k + 1] - 1e-12, 81)
                         for k in range(len(thk))])
    z6 = np.zeros(6)
    dE11 = -p ** 2 * Es
    s11 = np.empty_like(zc); s12 = np.empty_like(zc)
    for i, z in enumerate(zc):
        Sig = msgrm_strain_at_depth(r, z, Es, z6, z6, dE11, z6, z6)[1]
        s11[i] = Sig[0]; s12[i] = Sig[5]
    ze, sige, _, _ = ex.profile(n_per_layer=81)
    e11x = np.interp(zc, ze, sige[:, 0])
    e12x = np.interp(zc, ze, sige[:, 5])
    n11 = np.linalg.norm(e11x)
    e11 = 100 * np.linalg.norm(s11 - e11x) / n11
    e12 = 100 * np.linalg.norm(s12 - e12x) / max(np.linalg.norm(e12x),
                                                 1e-2 * n11)
    ratio12 = np.max(np.abs(s12)) / np.max(np.abs(s11))
    return e11, e12, ratio12, np.linalg.norm(e12x) / n11


lines = ["sigma_12 validation vs the exact Pagano solution",
         "=" * 60, "",
         "1. CROSS-PLY zero-field check (exact sigma_12 = 0 identically):",
         "   recovered max|s12|/max|s11| must sit at the round-off floor"]
print("\n".join(lines[-2:]))
for name, cfg in CROSS_CHECKS.items():
    e11, e12, ratio, scale = run_point(cfg["fr"], cfg["ang"], cfg["mats"],
                                       cfg["db"], cfg["S"])
    msg = ("   %-22s max|s12_rec|/max|s11_rec| = %.2e   (exact ||s12||/||s11||"
           " = %.1e)" % (name, ratio, scale))
    lines.append(msg)
    print(msg)

lines += ["", "2. ANGLE-PLY convergence sweep (Yu material, span a = 4 in,",
          "   errors = rel L2 vs exact at x = a/2; order between rows):"]
fig, ax = plt.subplots(figsize=(6.4, 5.0))
colors = {"[15/-15]": "#ff7f0e", "[30/-30/-30/30]": "#1f77b4"}
for name, cfg in LAYUPS_SWEEP.items():
    rows = []
    for S in S_LIST:
        e11, e12, ratio, scale = run_point(cfg["fr"], cfg["ang"], cfg["mats"],
                                           cfg["db"], S)
        rows.append((S, e11, e12))
    lines.append("   %s:" % name)
    lines.append("     S = a/h    s11 err%%   s12 err%%   order(s12)")
    print("  ", name)
    prev = None
    for S, e11, e12 in rows:
        order = ("  %5.2f" % (np.log(prev[2] / e12) / np.log(S / prev[0]))
                 if prev else "     --")
        ln = "     %-8g %9.4f %10.4f %s" % (S, e11, e12, order)
        lines.append(ln)
        print(ln)
        prev = (S, e11, e12)
    ax.loglog([r[0] for r in rows], [r[2] for r in rows], "o-",
              color=colors[name], label=r"$\sigma_{12}$  %s" % name)
    ax.loglog([r[0] for r in rows], [r[1] for r in rows], "s--", mfc="none",
              color=colors[name], label=r"$\sigma_{11}$  %s" % name)
Sarr = np.array(S_LIST)
ax.loglog(Sarr, 100 * (Sarr[0] / Sarr) ** 2 * 0.35, ":", color="0.5",
          label=r"$\mathcal{O}(S^{-2})$ guide")
ax.set_xlabel("S = a/h", fontsize=11)
ax.set_ylabel("rel. L2 error vs exact [%]", fontsize=11)
ax.grid(alpha=0.3, which="both")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
          fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "sig12_sweep.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

lines += ["", "verdict: a ~2nd-order falling error curve = the recovery is",
          "correct and the thick-plate numbers are thickness effects; a",
          "plateau would indicate an operator/driver defect."]
with open(os.path.join(HERE, "sig12_sweep.dat"), "w") as f:
    f.write("\n".join(lines) + "\n")
