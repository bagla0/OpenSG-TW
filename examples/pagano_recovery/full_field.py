"""full_field.py -- ALL SIX stress components and ALL THREE displacements of the
four curated Pagano cases, recovered by OpenSG-RM and compared with the exact
3-D solution in ONE figure per case (the VAPAS-grade full-field deliverable).

Recovery composition (the settled rules; see msg_rm_plate docstrings):
  sigma_11/22/12  x = a/2, Eq. 66: E6 = Es (RM measures), E6,11 = -p^2 Es,
                  detilted Gamma_l columns, + the face LOAD ladders
  sigma_13/23     x = 0, Eq. 66: E6,1 = p Es + the load-gradient ladders
  sigma_33        x = a/2, through-thickness EQUILIBRIUM integration of the
                  x = 0 sigma_13 amplitude from the loaded bottom face -- the
                  uniformly best route (the DIRECT Eq.-66 route with the load
                  columns has machine-exact faces and is recorded in the .dat,
                  but its interior wobbles for the soft-core sandwich)
  U_1, U_2        x = 0, Eq. 65 Kirchhoff composition U - z (p W) + w1 etc.,
                  load-gradient ladders in the warping
  U_3             x = a/2, W + w3 with the full load ladder (the pressure
                  compression through the thickness)
The plate part u^2d comes from the Abaqus RM-shell job when present
(--u2d abaqus, the default), else the internal harmonic solve.

Run:
    python examples/pagano_recovery/full_field.py [--case NAME] [--u2d theory]
Writes full_field.dat + full_field.png into each case's ORIGINAL folder.

Script variables
----------------
cf              the case_setup dict (+ qt/qb face-pressure fractions)
y, Es           plate DOF and strain amplitudes (Abaqus-fitted or theory)
qt6_m/qb6_m     face load ladders at the sin peak [q, 0, 0, -p^2 q, 0, 0]
qt6_e/qb6_e     the gradient-only ladders at x = 0 [0, p q, 0, 0, 0, 0]
S_m, S_e        recovered stress profiles at the two stations
s33_eq          the equilibrium-integrated sigma_33 (plotted); s33_dir = the
                direct Eq.-66 route (faces machine-exact, .dat only)
u1r/u2r/u3r     the recovered displacement profiles
ERR             the 9 (+1 direct-s33) rel-L2 errors vs exact; near-zero exact
                fields get the standard denominator floors and are flagged
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from recovery_bench import (CASES, case_setup, plate_dofs_theory,     # noqa: E402
                            fit_plate_dofs, harmonic_ops, _grid, _relerr)
from opensg_jax.fe_jax.msg_rm_plate import (msgrm_strain_at_depth,    # noqa: E402
                                            msgrm_warping_at_depth)


def run_case(name, u2d="abaqus"):
    """One case, all nine fields (variables in the module docstring)."""
    cf = case_setup(name)
    thk, p, h, a, q0 = cf["thk"], cf["p"], cf["h"], cf["a"], cf["q0"]
    zc = _grid(thk)
    B_E, _ = harmonic_ops(p)
    res = float("nan")
    if u2d == "abaqus" and os.path.isfile(cf["abq"]):
        y, res = fit_plate_dofs(cf["abq"], a)
        Es = B_E @ y
        src = "Abaqus RM shell (fit residual %.1e)" % res
    else:
        y, Es = plate_dofs_theory(cf["A6"], cf["G2"], p, q0)
        src = "internal harmonic solve"
        u2d = "theory"

    z6 = np.zeros(6)
    dE1 = p * Es
    dE11 = -p ** 2 * Es
    qt6_m = np.array([1, 0, 0, -p ** 2, 0, 0]) * (cf["qt"] * q0)
    qb6_m = np.array([1, 0, 0, -p ** 2, 0, 0]) * (cf["qb"] * q0)
    qt6_e = np.array([0, p, 0, 0, 0, 0]) * (cf["qt"] * q0)
    qb6_e = np.array([0, p, 0, 0, 0, 0]) * (cf["qb"] * q0)

    n = len(zc)
    S_m = np.empty((n, 6)); S_e = np.empty((n, 6))
    u1r = np.empty(n); u2r = np.empty(n); u3r = np.empty(n)
    for i, z in enumerate(zc):
        S_m[i] = msgrm_strain_at_depth(cf["r"], z, Es, z6, z6, dE11, z6, z6,
                                       qt6=qt6_m, qb6=qb6_m)[1]
        S_e[i] = msgrm_strain_at_depth(cf["r"], z, z6, dE1, z6, z6, z6, z6,
                                       qt6=qt6_e, qb6=qb6_e)[1]
        w0 = msgrm_warping_at_depth(cf["r"], z, z6, dE1, z6, z6, z6, z6,
                                    qt6=qt6_e, qb6=qb6_e)
        wm = msgrm_warping_at_depth(cf["r"], z, Es, z6, z6, dE11, z6, z6,
                                    qt6=qt6_m, qb6=qb6_m)
        u1r[i] = y[0] - z * p * y[2] + w0[0]
        u2r[i] = y[1] + w0[1]
        u3r[i] = y[2] + wm[2]
    # sigma_33: equilibrium integration of the x=0 s13 amplitude (families
    # s13 ~ cos -> s33 ~ sin), from the loaded bottom face
    s33_eq = cf["qb"] * q0 + np.concatenate(
        [[0.0], np.cumsum(0.5 * p * (S_e[1:, 4] + S_e[:-1, 4]) * np.diff(zc))])
    s33_dir = S_m[:, 2]

    ze, sige, _, uvw = cf["ex"].profile(n_per_layer=81)
    exs = np.column_stack([np.interp(zc, ze, sige[:, j]) for j in range(6)])
    exu = np.column_stack([np.interp(zc, ze, uvw[:, j]) for j in range(3)])

    n11 = np.linalg.norm(exs[:, 0]); nu1 = np.linalg.norm(exu[:, 0])
    models = {"s11": (S_m[:, 0], exs[:, 0], 0.0),
              "s22": (S_m[:, 1], exs[:, 1], 1e-2 * n11),
              "s12": (S_m[:, 5], exs[:, 5], 1e-2 * n11),
              "s13": (S_e[:, 4], exs[:, 4], 0.0),
              "s23": (S_e[:, 3], exs[:, 3], 1e-2 * np.linalg.norm(exs[:, 4])),
              "s33": (s33_eq, exs[:, 2], 0.0),
              "U1": (u1r, exu[:, 0], 0.0),
              "U2": (u2r, exu[:, 1], 1e-3 * nu1),
              "U3": (u3r, exu[:, 2], 0.0)}
    ERR = {}
    for k, (m, e, fl) in models.items():
        d = max(np.linalg.norm(e), fl) if fl else np.linalg.norm(e)
        ERR[k] = 100 * np.linalg.norm(m - e) / d
    e33dir = _relerr(s33_dir, exs[:, 2])

    hdr = ["%s -- FULL-FIELD recovery vs exact Pagano (all 6 stresses + 3 disp)"
           % cf["label"],
           "u^2d from: %s" % src,
           "stations: s11/s22/s12/s33/U3 at x = a/2; s13/s23/U1/U2 at x = 0",
           "face load ladders: qt = %g q0 (top), qb = %g q0 (bottom)"
           % (cf["qt"], cf["qb"]),
           "s33 = equilibrium integration (plotted); DIRECT Eq.-66 route with",
           "the load columns: faces machine-exact (bottom %.3g, top %.3g target"
           % (s33_dir[0], s33_dir[-1]),
           "  %.3g / %.3g), interior rel-L2 %.3f%%" % (cf["qb"] * q0,
                                                       cf["qt"] * q0, e33dir),
           "",
           "rel L2 errors vs exact [%] (near-zero exact fields floored):",
           "  " + "  ".join("%s %7.3f" % (k, ERR[k]) for k in
                            ("s11", "s22", "s12", "s13", "s23", "s33")),
           "  " + "  ".join("%s %7.3f" % (k, ERR[k]) for k in
                            ("U1", "U2", "U3")),
           "",
           "columns: z[%s]  then (rec, exact) pairs for s11 s22 s12 s13 s23 "
           "s33 U1 U2 U3" % cf["ulab"]]
    cols = [zc]
    for k in ("s11", "s22", "s12", "s13", "s23", "s33", "U1", "U2", "U3"):
        cols += [models[k][0], models[k][1]]
    np.savetxt(os.path.join(cf["outdir"], "full_field.dat"),
               np.column_stack(cols), header="\n".join(hdr), fmt="%15.6e")

    labels = [(r"$\sigma_{11}$", "x=a/2"), (r"$\sigma_{22}$", "x=a/2"),
              (r"$\sigma_{12}$", "x=a/2"), (r"$\sigma_{13}$", "x=0"),
              (r"$\sigma_{23}$", "x=0"), (r"$\sigma_{33}$", "x=a/2"),
              (r"$U_1$", "x=0"), (r"$U_2$", "x=0"), (r"$U_3$", "x=a/2")]
    keys = ("s11", "s22", "s12", "s13", "s23", "s33", "U1", "U2", "U3")
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 12.0))
    for ax, key, (sym, st) in zip(axes.ravel(), keys, labels):
        m, e, _ = models[key]
        ax.plot(e, zc / h, "-", color="k", lw=2.0,
                label="exact 3-D (Pagano)" if key == "s11" else None)
        ax.plot(m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2,
                lw=1.6, markevery=8,
                label="OpenSG-RM recovery" if key == "s11" else None)
        unit = cf["unit"] if key.startswith("s") else " [%s]" % cf["ulab"]
        ax.set_xlabel("%s%s  at  $%s$" % (sym, unit, st), fontsize=10)
        ax.grid(alpha=0.3)
    for r_ in range(3):
        axes[r_, 0].set_ylabel("$z/h$", fontsize=10)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(cf["outdir"], "full_field.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  %-14s " % name + "  ".join(
        "%s %6.2f" % (k, ERR[k]) for k in keys))
    return ERR


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--u2d", default="abaqus", choices=("abaqus", "theory"))
    args = ap.parse_args()
    print("full-field recovery (all 6 stresses + 3 displacements), rel L2 % vs exact")
    for nm in ([args.case] if args.case else list(CASES)):
        run_case(nm, u2d=args.u2d)
