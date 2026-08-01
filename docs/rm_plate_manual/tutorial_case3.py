"""OpenSG-RM Tutorial 3 -- antisymmetric angle-ply [15/-15] plate, L/h = 4.

Standalone: run it from anywhere inside the repository and it produces
    tutorial_case3_sg.yaml / _sg.png      the 1-D SG and its mesh figure
    tutorial_case3_outofplane.png         sigma_13, sigma_23, sigma_33
    tutorial_case3_inplane.png            sigma_11, sigma_22, sigma_12
    tutorial_case3_disp.png               U1, U2, U3
and prints the plate law, the recovery errors and the failure indices.

User options (PreVABS-style):
    --model 0    classical ABD: the 6x6 lamination matrix, Kirchhoff plate
                 kinematics (no transverse-shear refinement)
    --model 1    RM shear-refined 8x8 (default)
    --FF N11 N22 N12 M11 M22 M12 Q1 Q2
                 dehomogenize from YOUR resultants instead of the built-in
                 closed-form plate solution; the exact-3-D comparison is
                 skipped (an arbitrary FF has no exact twin)

Problem (default run): simply supported strip, span a = 4 in along x1,
thickness h = 1 in (DELIBERATELY THICK, L/h = 4), loaded on BOTH faces:
sigma_33 = +q0/2 sin(pi x/a) on the top and -q0/2 sin(pi x/a) on the
bottom, q0 = 1 psi.  Two equal plies [15/-15] bottom-up (25:1
stiffness-ratio material, psi-in units), mid-surface reference.  The
off-axis plies COUPLE: the plate solution carries nonzero N22, M22, M12,
Q2 and the response has sigma_23 and U2 -- all recovered below.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo-root discovery: works from any working directory in the repo ------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "garg"))   # exact 3-D reference

from opensg_jax.fe_jax.segment_plate import plate_sg_yaml, read_plate_sg_yaml, \
    plot_plate_sg
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth, \
    msgrm_warping_at_depth
from opensg_jax.fe_jax.msg_materials import rotation_6x6
from pagano_exact import ExactCyl                    # exact 3-D reference only

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", type=int, choices=(0, 1), default=1,
                help="0 = classical ABD, 1 = RM shear-refined (default)")
ap.add_argument("--FF", nargs=8, type=float, default=None,
                metavar=("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2"),
                help="dehomogenize from these resultants (skips the exact "
                     "comparison)")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# 1. material + layup  ->  1-D SG YAML  ->  homogenization
# ---------------------------------------------------------------------------
MATERIAL_DB = {"yu": {"E": [25.0e6, 1.0e6, 1.0e6],
                      "G": [0.5e6, 0.5e6, 0.2e6],
                      "nu": [0.25, 0.25, 0.25], "rho": 1.0}}
a, q0 = 4.0, 1.0
h = 1.0
layup = {"mat_names": ["yu", "yu"],
         "thick":     [0.5, 0.5],
         "angles":    [15.0, -15.0]}                 # bottom ply first

yml = os.path.join(HERE, "tutorial_case3_sg.yaml")
plate_sg_yaml(yml, layup, MATERIAL_DB, fraction=0.5)
plot_plate_sg(yml)                                    # writes ..._sg.png
inp = read_plate_sg_yaml(yml)
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])

A6 = np.asarray(r["A6"]); G2 = np.asarray(r["ABDG"])[6:8, 6:8]
ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")
if args.model == 0:
    print("model 0: CLASSICAL ABD (6x6; rows/cols: %s):" % ", ".join(ROWS[:6]))
    for name, row in zip(ROWS[:6], A6):
        print("%5s " % name + " ".join("%12.4e" % v for v in row))
else:
    print("model 1: RM SHEAR-REFINED 8x8 ABDG (rows/cols: %s):" % ", ".join(ROWS))
    for name, row in zip(ROWS, np.asarray(r["ABDG"])):
        print("%5s " % name + " ".join("%12.4e" % v for v in row))

# ---------------------------------------------------------------------------
# 2. the plate state driving the recovery: either the USER's --FF, or the
#    built-in single-harmonic closed form.  The 5-DOF solve supplies the
#    coupling resultants (N22, M22, M12, Q2) the off-axis plies generate --
#    statics alone fixes only Q1 and M11.
# ---------------------------------------------------------------------------
p = np.pi / a
S6 = np.linalg.inv(A6)
z6 = np.zeros(6)
if args.FF is not None:
    FF = np.asarray(args.FF, float)
    Es = S6 @ FF[:6]
    dE1 = S6 @ np.array([0, 0, 0, FF[6], 0, 0.0])       # dM11/dx = Q1
    dE2 = S6 @ np.array([0, 0, 0, 0, FF[7], 0.0])       # dM22/dy = Q2
    dE11 = z6
    qt_e = qb_e = qt_m = qb_m = None
    y5 = np.zeros(5)
    compare_exact = False
    print("\ndehomogenizing from user FF =", FF.tolist())
elif args.model == 1:                      # RM closed form of the strip
    BE = np.zeros((6, 5)); BE[0, 0] = BE[2, 1] = BE[3, 3] = BE[5, 4] = -p
    Bg = np.zeros((2, 5)); Bg[0, 2] = p; Bg[0, 3] = 1.0; Bg[1, 4] = 1.0
    K5 = BE.T @ A6 @ BE + Bg.T @ G2 @ Bg
    y5 = np.linalg.solve(K5, np.array([0, 0, q0, 0, 0.0]))  # [U, V, W, F1, F2]
    Es = BE @ y5
    dE1, dE2, dE11 = p * Es, z6, -p ** 2 * Es
    qt_e = np.array([0, p, 0, 0, 0, 0.0]) * (+q0 / 2)   # split face load
    qb_e = np.array([0, p, 0, 0, 0, 0.0]) * (-q0 / 2)
    qt_m = np.array([1, 0, 0, -p ** 2, 0, 0.0]) * (+q0 / 2)
    qb_m = np.array([1, 0, 0, -p ** 2, 0, 0.0]) * (-q0 / 2)
    compare_exact = True
    R6 = A6 @ Es
    print("\nresultant amplitudes [N11, N22, N12, M11, M22, M12] = "
          + "[" + ", ".join("%.6g" % v for v in R6) + "]")
    print("coupling shear Q2 = %.6g" % (G2 @ (Bg @ y5))[1])
else:                                      # classical (Kirchhoff) closed form
    idx = [0, 2, 3]                        # unknowns U, V, W; k11 = +p^2 W
    B3 = np.zeros((6, 3)); B3[0, 0] = B3[2, 1] = -p; B3[3, 2] = p ** 2
    K3 = (A6 @ B3)[idx, :]
    uvw = np.linalg.solve(K3, np.array([0, 0, q0 / p ** 2]))
    y5 = np.array([uvw[0], uvw[1], uvw[2], -p * uvw[2], 0.0])
    Es = B3 @ uvw
    dE1, dE2, dE11 = p * Es, z6, z6
    qt_e = qb_e = qt_m = qb_m = None
    compare_exact = True
print("plate solution amplitudes [U, V, W, F1, F2] = "
      + "[" + ", ".join("%.6g" % v for v in y5) + "]")
print("check M11 amplitude = %.6g  (statics q0/p^2 = %.6g)"
      % ((A6 @ Es)[3], q0 / p ** 2))

# ---------------------------------------------------------------------------
# 3. dehomogenization through the thickness at the two stations; the
#    sigma_33 equilibrium integration starts from the LOADED bottom face
# ---------------------------------------------------------------------------
zc = np.linspace(-h / 2 + 1e-9, h / 2 - 1e-9, 201)
S_end = np.array([msgrm_strain_at_depth(r, z, z6, dE1, dE2,
                                        qt6=qt_e, qb6=qb_e)[1]
                  for z in zc])                            # x = 0 station
S_mid = np.array([msgrm_strain_at_depth(r, z, Es, dE11=dE11,
                                        qt6=qt_m, qb6=qb_m)[1]
                  for z in zc])                            # x = a/2 station
s33_0 = -q0 / 2 if args.FF is None and args.model == 1 else 0.0
s33 = s33_0 + np.concatenate(
    [[0.0], np.cumsum(0.5 * p * (S_end[1:, 4] + S_end[:-1, 4]) * np.diff(zc))])
U = np.empty((len(zc), 3))
for i, z in enumerate(zc):
    w0 = msgrm_warping_at_depth(r, z, z6, dE1, dE2, qt6=qt_e, qb6=qb_e)
    wm = msgrm_warping_at_depth(r, z, Es, dE11=dE11, qt6=qt_m, qb6=qb_m)
    U[i] = [y5[0] - z * p * y5[2] + w0[0],     # U1 = u - z w,1 + warping
            y5[1] + w0[1],                     # U2 (w,2 = 0)
            y5[2] + wm[2]]                     # U3 = w + thickness warping

if compare_exact:
    ex = ExactCyl(layup["thick"], layup["angles"], layup["mat_names"],
                  MATERIAL_DB, a, q0=q0 / 2, q_bot=-q0 / 2)
    ze, sige, _, uvwe = ex.profile(n_per_layer=81)

    def rel(m, e, floor=0.0):
        return 100 * np.linalg.norm(m - np.interp(zc, ze, e)) / \
            max(np.linalg.norm(np.interp(zc, ze, e)), floor)

    print("\nrecovery errors vs exact 3-D (rel L2 %):")
    print("  s13 %6.2f   s23 %6.2f   s33 %6.2f"
          % (rel(S_end[:, 4], sige[:, 4]), rel(S_end[:, 3], sige[:, 3]),
             rel(s33, sige[:, 2])))
    print("  s11 %6.2f   s22 %6.2f   s12 %6.2f"
          % (rel(S_mid[:, 0], sige[:, 0]), rel(S_mid[:, 1], sige[:, 1]),
             rel(S_mid[:, 5], sige[:, 5])))
    print("  U1  %6.2f   U2  %6.2f   U3  %6.2f"
          % (rel(U[:, 0], uvwe[:, 0]), rel(U[:, 1], uvwe[:, 1]),
             rel(U[:, 2], uvwe[:, 2])))

# ---------------------------------------------------------------------------
# 4. figures: OpenSG-RM (orange); exact 3-D (black) when comparable
# ---------------------------------------------------------------------------
def panel(ax, m, e, xlabel, label=False):
    if compare_exact:
        ax.plot(e, ze / h, "-", color="k", lw=2.0,
                label="exact 3-D" if label else None)
    ax.plot(m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2,
            lw=1.6, markevery=10, label="OpenSG-RM" if label else None)
    ax.set_xlabel(xlabel, fontsize=11); ax.grid(alpha=0.3)


exs = sige if compare_exact else np.zeros((1, 6))
exu = uvwe if compare_exact else np.zeros((1, 3))
for fname, specs in (
    ("outofplane",
     [(S_end[:, 4], exs[:, 4], r"$\sigma_{13}$ [psi] at $x=0$"),
      (S_end[:, 3], exs[:, 3], r"$\sigma_{23}$ [psi] at $x=0$"),
      (s33, exs[:, 2], r"$\sigma_{33}$ [psi] at $x=a/2$")]),
    ("inplane",
     [(S_mid[:, 0], exs[:, 0], r"$\sigma_{11}$ [psi] at $x=a/2$"),
      (S_mid[:, 1], exs[:, 1], r"$\sigma_{22}$ [psi] at $x=a/2$"),
      (S_mid[:, 5], exs[:, 5], r"$\sigma_{12}$ [psi] at $x=a/2$")]),
    ("disp",
     [(U[:, 0], exu[:, 0], r"$U_1$ [in] at $x=0$"),
      (U[:, 1], exu[:, 1], r"$U_2$ [in] at $x=0$"),
      (U[:, 2], exu[:, 2], r"$U_3$ [in] at $x=a/2$")])):
    fig, axes = plt.subplots(1, len(specs), figsize=(4.4 * len(specs), 4.6))
    for k, (m, e, lbl) in enumerate(specs):
        panel(np.atleast_1d(axes)[k], m, e, lbl, label=(k == 0))
    np.atleast_1d(axes)[0].set_ylabel("$z/h$", fontsize=11)
    hnd, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "tutorial_case3_%s.png" % fname), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
print("\nwrote tutorial_case3_{sg,outofplane,inplane,disp}.png")

# ---------------------------------------------------------------------------
# 5. failure indices from the recovered ply-frame stress (BOTH criteria)
#    Allowables below are EXAMPLE values in psi -- replace with yours.
# ---------------------------------------------------------------------------
Xt, Xc = 150e3, 100e3           # fiber tension / compression [psi]
Yt, Yc = 6e3, 17e3              # transverse tension / compression
S12a, S13a, S23a = 10e3, 9e3, 9e3
F1 = 1/Xt - 1/Xc; F2 = 1/Yt - 1/Yc
F11 = 1/(Xt*Xc); F22 = 1/(Yt*Yc); F66 = 1/S12a**2
F44 = 1/S23a**2; F55 = 1/S13a**2
F12 = -0.5 * np.sqrt(F11 * F22)

ms_worst = tw_worst = 0.0
for z, Sig in [(z, S) for z, S in zip(zc, S_end)] + \
              [(z, S) for z, S in zip(zc, S_mid)]:
    th = msgrm_strain_at_depth(r, z, z6)[2]
    sm = rotation_6x6(-th) @ Sig                 # ply frame [11,22,33,23,13,12]
    s1, s2, s23_, s13_, s12_ = sm[0], sm[1], sm[3], sm[4], sm[5]
    ms = max(s1/Xt if s1 > 0 else -s1/Xc, s2/Yt if s2 > 0 else -s2/Yc,
             abs(s12_)/S12a, abs(s13_)/S13a, abs(s23_)/S23a)
    tw = (F1*s1 + F2*s2 + F11*s1**2 + F22*s2**2 + F66*s12_**2
          + F44*s23_**2 + F55*s13_**2 + 2*F12*s1*s2)
    ms_worst = max(ms_worst, ms); tw_worst = max(tw_worst, tw)
print("\nfailure indices over both stations (example allowables):")
print("  max-stress index = %.4e   Tsai-Wu index = %.4e   (>= 1 = failure)"
      % (ms_worst, tw_worst))
