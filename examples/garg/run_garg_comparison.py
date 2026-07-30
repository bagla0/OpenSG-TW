"""run_garg_comparison.py -- the Garg et al. (2023) benchmark through the CORE MSG-RM
module, 1D-yaml route end to end.

Garg train a GPR surrogate on 3-D elasticity solutions because FSDT cannot produce the
through-thickness distributions (sigma13 piecewise constant and face-violating; sigma33
absent).  Here the SAME configurations are answered analytically:

  garg_plates.yaml -> msg_mesh.load_yaml -> core rm_plate_msg (homo; quartic 5-noded)
  -> core msgrm_strain_at_depth (Eq. 63/66 recovery; V2 active via dE11 = -p^2 E)
  -> sigma33 from through-thickness equilibrium of the recovered sigma13
  vs the EXACT 3-D elasticity solution (exact_cyl.ExactCyl, cylindrical bending,
  sigma33(x, top) = q0 sin(p x)), with FSDT (Garg's baseline) alongside.

Outputs: garg_results.dat (relative L2 errors + peaks for every case/S), and per-case
figures garg_<case>_S<val>.png (sigma11 / sigma13 / sigma33 through the thickness).
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
sys.path.insert(0, os.path.join(CC, "examples", "TW-paper", "rm_thickness"))

from opensg_jax.fe_jax.msg_mesh import load_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from exact_cyl import ExactCyl
import cyl_models as CM                          # FSDT baseline + z-grid conventions

YAML = os.path.join(HERE, "garg_plates.yaml")
_, _, MDB, LDB, _ = load_yaml(YAML)

S_OF = {"garg_A_090": (100, 10, 5, 4), "garg_A_909": (100, 5),
        "garg_B_0990": (10, 5), "garg_B_0909": (10, 5),
        "garg_C_sand": (20, 10, 5, 4)}
FIGS = {("garg_A_090", 4), ("garg_B_0990", 5), ("garg_C_sand", 4)}
NPLY_PROF = 61                                   # samples per ply for the profiles
EXACTC, MSGC, FSDTC = "k", "#ff7f0e", "#1f77b4"


def relerr(a, b):
    b = np.asarray(b, float)
    return float(np.linalg.norm(np.asarray(a, float) - b) / (np.linalg.norm(b) + 1e-300))


def integ(z, f):
    F = np.zeros_like(f)
    F[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(z))
    return F


rep = ["# Garg et al. (2023) benchmark vs core MSG-RM (1D-yaml route: garg_plates.yaml)",
       "# exact = 3-D elasticity (cylindrical bending, q = q0 sin(p x)); FSDT = Garg's baseline",
       "#   (Garg's GPR is TRAINED on these same exact solutions; MSG-RM reaches them",
       "#    analytically, no surrogate, no training data)",
       "# MSG-RM     : consistent FIRST-order recovery (in-plane + sigma13 constitutive,",
       "#              sigma33 by through-thickness equilibrium) -- the primary model",
       "# MSG-RM-2nd : Eq. (50) classical measures + Eq. (66) gradient terms (V2 active);",
       "#              INCOMPLETE without the V1L/V2L load columns for a pressure-loaded",
       "#              plate: ~20x better than first order in the thin regime, diverges",
       "#              thick -- reported to document exactly that",
       "#",
       "# %-12s %5s %-10s %10s %10s %10s   %10s %10s" %
       ("case", "S", "model", "err_s11%", "err_s13%", "err_s33%", "peak_s13", "peak_s33")]

for name, lay in LDB.items():
    thk = [float(t) for t in lay["thick"]]
    ang = [float(a) for a in lay["angles"]]
    mats = [str(m) for m in lay["mat_names"]]
    h = float(sum(thk))
    r = rm_plate_msg(thk, ang, mats, MDB, fraction=0.5)          # CORE homo (quartic)
    obj = dict(r); obj["thick"] = np.asarray(thk); obj["z_ref"] = 0.5 * h   # FSDT adapter

    from opensg_jax.fe_jax.msg_transverse_shear import plate_8x8
    P8 = plate_8x8(np.asarray(r["A6"]), np.asarray(r["G_msg"]))
    rep.append("# %s: 8x8 ABDG (core rm_plate_msg, mid reference; rows e11,e22,g12,"
               "k11,k22,k12,2g13,2g23):" % name)
    for row in P8:
        rep.append("#   " + " ".join("%13.5e" % v for v in row))

    for S in S_OF[name]:
        L = S * h
        ex = ExactCyl(thk, ang, mats, MDB, L, q0=1.0)
        p = ex.p
        zc, sig_e, _, _ = ex.profile(n_per_layer=NPLY_PROF)

        E6 = CM.plate_strains(np.asarray(r["A6"]), p, q0=1.0)
        fs = CM.fsdt_profile(obj, E6, p, q0=1.0, n_per_layer=NPLY_PROF)
        assert np.allclose(zc, fs["z"], atol=1e-9 * h)

        # ---- CORE recovery, FIRST order (the consistent truncation without the load
        #      columns; = the rm-thickness paper's msg_profile) ----
        s_m = np.array([msgrm_strain_at_depth(r, z, E6)[1] for z in zc])
        s_s = np.array([msgrm_strain_at_depth(r, z, np.zeros(6), p * E6)[1] for z in zc])
        s13 = s_s[:, 4]; s23 = s_s[:, 3]
        s33 = integ(zc, p * s13)                                  # d s33/dz = p * s13

        # ---- SECOND order: Eq. (50) classical measures (eps = R - D1 gamma,1 with
        #      gamma = G^-1 [q0/p, 0] cos) + the Eq. (66) gradient terms (V2 active).
        #      NOTE: without the V1L/V2L load columns this is an INCOMPLETE second order
        #      for a pressure-loaded plate -- superb in the thin regime, diverges thick.
        D1s = np.zeros((6, 2)); D1s[3, 0] = 1.0; D1s[5, 1] = 1.0
        gam = np.linalg.solve(np.asarray(r["G_msg"]), np.array([1.0 / p, 0.0]))
        E6c = E6 + p * (D1s @ gam)
        s_m2 = np.array([msgrm_strain_at_depth(r, z, E6c, dE11=-p * p * E6c)[1] for z in zc])
        s_s2 = np.array([msgrm_strain_at_depth(r, z, np.zeros(6), p * E6c)[1] for z in zc])
        s13_2 = s_s2[:, 4]
        s33_2 = integ(zc, p * s13_2)

        i13 = np.argmax(np.abs(sig_e[:, 4])); i33 = np.argmax(np.abs(sig_e[:, 2]))
        rows = [("FSDT", relerr(fs["s11"], sig_e[:, 0]), relerr(fs["s13"], sig_e[:, 4]),
                 np.nan, fs["s13"][i13], np.nan),
                ("MSG-RM", relerr(s_m[:, 0], sig_e[:, 0]), relerr(s13, sig_e[:, 4]),
                 relerr(s33, sig_e[:, 2]), s13[i13], s33[i33]),
                ("MSG-RM-2nd", relerr(s_m2[:, 0], sig_e[:, 0]), relerr(s13_2, sig_e[:, 4]),
                 relerr(s33_2, sig_e[:, 2]), s13_2[i13], s33_2[i33])]
        rep.append("# exact peaks     %-5s S=%-4d  s13 %10.4f   s33 %10.4f"
                   % (name.split("_")[1], S, sig_e[i13, 4], sig_e[i33, 2]))
        for tag, e11, e13, e33, p13, p33 in rows:
            rep.append("%-14s %5d %-10s %10.3f %10.3f %10s   %10.4f %10s"
                       % (name, S, tag, 100 * e11, 100 * e13,
                          ("%10.3f" % (100 * e33)) if np.isfinite(e33) else "       n/a",
                          p13, ("%10.4f" % p33) if np.isfinite(p33) else "       n/a"))
        print(rep[-3]); print(rep[-2], flush=True)

        if (name, S) in FIGS:
            zn = zc / h
            fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
            panels = [(0, r"$\sigma_{11}/q_0$", s_m[:, 0], fs["s11"]),
                      (4, r"$\sigma_{13}/q_0$", s13, fs["s13"]),
                      (2, r"$\sigma_{33}/q_0$", s33, None)]
            for ax, (ci, lab, msg_v, fsdt_v) in zip(axs, panels):
                ax.plot(sig_e[:, ci], zn, "-", color=EXACTC, lw=2.0, label="exact 3-D")
                ax.plot(msg_v, zn, "--s", color=MSGC, ms=3, mfc="none", mew=1.1, lw=1.4,
                        markevery=7, label="MSG-RM (core)")
                if fsdt_v is not None:
                    ax.plot(fsdt_v, zn, ":", color=FSDTC, lw=1.8, label="FSDT")
                else:
                    ax.text(0.03, 0.03, "FSDT: n/a", transform=ax.transAxes,
                            fontsize=9, color=FSDTC)
                ax.set_xlabel(lab, fontsize=12); ax.set_ylabel(r"$z/h$", fontsize=12)
                ax.grid(alpha=0.3)
            axs[0].legend(fontsize=10, frameon=False, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(HERE, "%s_S%d.png" % (name, S)),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

# ---- headline summary table (FSDT vs MSG-RM, the sigma13/sigma33 story) ----
summ = ["#", "# ============== HEADLINE SUMMARY (rel L2 err vs exact 3-D) ==============",
        "# %-14s %5s | %10s %10s | %10s %10s" %
        ("case", "S", "s13 FSDT", "s13 MSG", "s33 FSDT", "s33 MSG")]
cur = {}
for ln in rep:
    t = ln.split()
    if len(t) >= 6 and not ln.startswith("#") and t[2] in ("FSDT", "MSG-RM"):
        cur.setdefault((t[0], t[1]), {})[t[2]] = (t[4], t[5])
for (case, S), d_ in cur.items():
    if "FSDT" in d_ and "MSG-RM" in d_:
        summ.append("# %-14s %5s | %10s %10s | %10s %10s"
                    % (case, S, d_["FSDT"][0], d_["MSG-RM"][0],
                       d_["FSDT"][1], d_["MSG-RM"][1]))
rep = summ + ["#"] + rep

open(os.path.join(HERE, "garg_results.dat"), "w").write("\n".join(rep) + "\n")
print("wrote garg_results.dat + figures", flush=True)
