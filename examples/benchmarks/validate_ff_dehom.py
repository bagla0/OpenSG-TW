"""Validate the FF-driven dehom (example 7 process) against EXACT 3-D elasticity.

Setup: Pagano [0/90/0] under cylindrical bending q = q0 sin(p x), the exact solution
from exact_cyl.ExactCyl.  At the section x0 = a/4 (both M and Q nonzero there):

  1. integrate the EXACT stresses through the thickness -> the section resultants
     N11, N22, M11, M22, Q1  (this is the 8x1 FF a user would supply)
  2. run the example-7 process on that FF with THREE gradient closures:
       A  dE1 = S6 @ [0,0,0, Q1, 0, 0]          (pure-M11 equilibrium closure, current)
       B  dE1 = S6 @ [0,0,0, Q1, Q1*M22/M11, 0] (proportional moments: all resultants
                                                  share one x-variation, M,1 ~ M * Q1/M11)
       C  dE1 = exact plate-strain gradient p*cot(p x0) * E6   (the true one, oracle)
  3. compare the recovered sigma13(z) and sigma11(z) with the exact 3-D profile at x0.
"""
import os, sys
import numpy as np

ROOT = os.path.expanduser("~/OpenSG-TW-claude")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "TW-paper", "rm_thickness"))
import jax
jax.config.update("jax_enable_x64", True)

from exact_cyl import ExactCyl
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth

MDB = {"pagano": {"E": [25.0e9, 1.0e9, 1.0e9], "G": [0.5e9, 0.5e9, 0.2e9],
                  "nu": [0.25, 0.25, 0.25], "rho": 1.0}}

q0 = 1.0e4; a = 1.0
for S in (10, 4):
    H = a / S
    thk = [H / 3] * 3; ang = [0.0, 90.0, 0.0]; mats = ["pagano"] * 3
    ex = ExactCyl(thk, ang, mats, MDB, a, q0=q0)
    p = np.pi / a
    x0 = 0.25 * a                                     # sin, cos both ~ 0.707
    sx0, cx0 = np.sin(p * x0), np.cos(p * x0)

    # dense exact profile at the section (amplitudes -> values at x0)
    zc, sig_e, eps_e, uvw_e = ex.profile(n_per_layer=161)
    # families: sigma11, sigma22, sigma33 ~ sin(px); sigma13 ~ cos(px)
    s11_ex = sig_e[:, 0] * sx0
    s13_ex = sig_e[:, 4] * cx0

    # 1 ---- exact section resultants at x0 (the user's FF)
    N11 = np.trapezoid(sig_e[:, 0], zc) * sx0
    N22 = np.trapezoid(sig_e[:, 1], zc) * sx0
    M11 = np.trapezoid(sig_e[:, 0] * zc, zc) * sx0
    M22 = np.trapezoid(sig_e[:, 1] * zc, zc) * sx0
    Q1 = np.trapezoid(sig_e[:, 4], zc) * cx0
    FF6 = np.array([N11, N22, 0.0, M11, M22, 0.0])
    print("=" * 86)
    print("S = %d   x0 = a/4:  N11 %.4e  N22 %.4e  M11 %.4e  M22 %.4e  Q1 %.4e"
          % (S, N11, N22, M11, M22, Q1))
    print("  equilibrium check dM11/dx = p*cot(px0)*M11 = %.4e  vs Q1 = %.4e  (%.3f%%)"
          % (p * M11 * cx0 / sx0, Q1, 100 * abs(p * M11 * cx0 / sx0 / Q1 - 1)))

    # 2 ---- homogenize + the three closures
    r = rm_plate_msg(thk, ang, mats, MDB, fraction=0.5)
    S6 = np.linalg.inv(r["A6"])
    E6 = S6 @ FF6
    dE1_A = S6 @ np.array([0, 0, 0, Q1, 0, 0.0])
    dE1_B = S6 @ np.array([0, 0, 0, Q1, Q1 * M22 / M11, 0.0])
    # oracle: every plate field ~ sin(px) -> E,1(x0) = p cos/sin * E(x0)
    dE1_C = p * (cx0 / sx0) * E6
    z6 = np.zeros(6)

    print("  E6      = %s" % np.array2string(E6, precision=3))
    print("  dE1  A  = %s" % np.array2string(dE1_A, precision=3))
    print("       B  = %s" % np.array2string(dE1_B, precision=3))
    print("       C  = %s" % np.array2string(dE1_C, precision=3))

    # 3 ---- recover and compare
    prof = {}
    for tag, dE1 in (("A", dE1_A), ("B", dE1_B), ("C", dE1_C)):
        s11 = np.empty_like(zc); s13 = np.empty_like(zc)
        for i, z in enumerate(zc):
            _, Sig, _ = msgrm_strain_at_depth(r, z, E6, dE1, z6)
            s11[i], s13[i] = Sig[0], Sig[4]
        prof[tag] = (s11, s13)
        e11 = np.linalg.norm(s11 - s11_ex) / np.linalg.norm(s11_ex)
        e13 = np.linalg.norm(s13 - s13_ex) / np.linalg.norm(s13_ex)
        Q1r = np.trapezoid(s13, zc)
        print("  closure %s:  s11 rel err %7.3f%%   s13 rel err %7.3f%%   int s13 = %.4e (Q1 %.4e)"
              % (tag, 100 * e11, 100 * e13, Q1r, Q1))
    dAB = np.linalg.norm(prof["A"][1] - prof["B"][1]) / np.linalg.norm(prof["B"][1])
    dBC = np.linalg.norm(prof["B"][1] - prof["C"][1]) / np.linalg.norm(prof["C"][1])
    print("  s13 profile difference:  A vs B %.3f%%   B vs C %.4f%%" % (100 * dAB, 100 * dBC))
