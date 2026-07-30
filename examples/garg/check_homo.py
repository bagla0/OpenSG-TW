"""check_homo.py -- validate the HOMOGENIZATION (the 8x8 ABDG) of the three Garg
laminates against the theories that carry no homogenization error of their own.

Load case (Garg sec. 3, Pagano's classic problem): CYLINDRICAL BENDING -- an infinite
simply-supported plate strip of span a under the sinusoidal top-face pressure

    q(x1) = q0 sin(p x1),   p = pi/a

Nothing varies with x2, so the plate response involves only (e11, k11, gamma13) and,
for a symmetric laminate, the section law collapses to two numbers:

    D11 = ABDG[3,3]   bending stiffness      (M11 = D11 k11 under the constraint)
    G11 = ABDG[6,6]   transverse shear stiffness

and the plate's mid-span deflection amplitude has the closed form

    w_plate = q0 / (p^4 D11)  +  q0 / (p^2 G11)          (bending + shear deflection)

The EXACT 3-D elasticity solution (exact_cyl.ExactCyl, the state-space Pagano
solution) gives w with no plate assumption at all, so comparing w_plate against it
tests BOTH homogenized stiffnesses at once: D11 dominates when the plate is thin,
G11 becomes a large fraction of w as S = a/h drops.  The same formula with Garg's
baseline shear stiffness G_fsdt = (5/6) sum(t_k C55_k) shows what the k = 5/6
assumption costs.  A6 is also checked against classical lamination (compute_ABD).

Run:
    python examples/garg/check_homo.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CC)
sys.path.insert(0, os.path.join(CC, "examples", "TW-paper", "rm_thickness"))

from exact_cyl import ExactCyl
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg
from opensg_jax.fe_jax.msg_materials import compute_ABD_matrix, rotated_stiffness_6x6
from garg_layups import MATERIAL_DB, LAYUPS, H

q0 = 1.0e4
a = 1.0
p = np.pi / a
ASPECTS = (4, 10, 20, 100)

print("cylindrical bending  q = q0 sin(p x),  mid-span deflection amplitude w")
print("w_plate = q0/(p^4 D11) + q0/(p^2 G11);  exact = 3-D elasticity (Pagano)")
print("=" * 84)
print("%-6s %5s | %12s | %12s %8s | %12s %8s | %7s" %
      ("case", "S", "w exact", "w MSG", "%err", "w FSDT", "%err", "shear%"))
for name, lay0 in LAYUPS.items():
    fr = [t / H for t in lay0["thick"]]           # the layup FRACTIONS define the family
    ang = lay0["angles"]; mats = lay0["mat_names"]
    for S in ASPECTS:
        h = a / S
        thk = [f * h for f in fr]

        r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
        D11 = float(r["ABDG"][3, 3]); G11 = float(r["ABDG"][6, 6])

        # internal consistency: A6 must be classical lamination theory exactly
        A_ref = np.asarray(compute_ABD_matrix(thk, ang, mats, MATERIAL_DB,
                                              n_per_layer=4, z_ref=h / 2)[0])[:6, :6]
        assert np.max(np.abs(r["A6"] - A_ref)) < 1e-9 * np.max(np.abs(A_ref))

        # Garg's baseline shear stiffness: k = 5/6, C55 of each (rotated) ply
        G_fsdt = 5.0 / 6.0 * sum(
            t * float(rotated_stiffness_6x6(MATERIAL_DB[m]["E"], MATERIAL_DB[m]["G"],
                                            MATERIAL_DB[m]["nu"], x)[4, 4])
            for t, m, x in zip(thk, mats, ang))

        w_msg = q0 / (p ** 4 * D11) + q0 / (p ** 2 * G11)
        w_fsdt = q0 / (p ** 4 * D11) + q0 / (p ** 2 * G_fsdt)

        ex = ExactCyl(thk, ang, mats, MATERIAL_DB, a, q0=q0)
        zc, _, _, uvw = ex.profile(n_per_layer=81)
        w_ex = float(uvw[np.argmin(np.abs(zc)), 2])   # mid-plane amplitude

        shear_frac = (q0 / (p ** 2 * G11)) / w_msg
        print("%-6s %5d | %12.5e | %12.5e %+7.2f%% | %12.5e %+7.2f%% | %6.1f%%" %
              (name, S, w_ex, w_msg, 100 * (w_msg / w_ex - 1),
               w_fsdt, 100 * (w_fsdt / w_ex - 1), 100 * shear_frac))
    print("-" * 84)
