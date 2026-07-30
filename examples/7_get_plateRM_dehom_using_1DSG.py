"""Example 7 - MSG-RM plate dehomogenization from a 1-D SG mesh YAML, driven by the
plate FORCE RESULTANTS (the plate's FF vector), like the beam dehomogenization.

Reads the same through-thickness 1-D SG as example 6, homogenizes (the 8x8 ABDG), and
recovers the through-thickness 3-D stress for an applied resultant vector

    FF = [N11, N22, N12, M11, M22, M12, Q1, Q2]     (forces N/m, moments N, shears N/m)

exactly as the beam dehomogenization takes the Timoshenko force-moment resultants:

  global plate strains  [E; gamma] = inv(ABDG) @ FF        (the constitutive inversion)
  strain gradients      dE/dx_a = S6 @ dFF6/dx_a with the moment gradients set by plate
                        EQUILIBRIUM  M11,1 = Q1 and M22,2 = Q2  (N and Q constant, the
                        plate analog of the beam's constant-shear state dM/dx = Q)
  recovery              core msgrm_strain_at_depth, Eq. (63)

The printed resultant check re-integrates the recovered stress through the thickness --
N = int sigma dz, M = int sigma z dz, Q = int tau dz -- and compares it to the input FF.

Run:
    python examples/7_get_plateRM_dehom_using_1DSG.py                       # unit M11
    python examples/7_get_plateRM_dehom_using_1DSG.py --FF 0 0 0 1e3 0 0 0 0
    python examples/7_get_plateRM_dehom_using_1DSG.py --FF 0 0 0 0 0 0 1e3 0   # pure Q1
    python examples/7_get_plateRM_dehom_using_1DSG.py --yaml my_plate_sg.yaml --out r.dat
"""
import argparse
import os
import sys

import numpy as np

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
np.set_printoptions(precision=4, linewidth=150)

from opensg_jax.fe_jax.segment_plate import read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth

FLBL = ["N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2"]
SLBL = ["s11", "s22", "s33", "s23", "s13", "s12"]
DEFAULT_YAML = os.path.join(CC, "examples", "data", "1d_yaml", "plate_sym45_sg.yaml")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--yaml", default=DEFAULT_YAML, help="through-thickness plate 1-D SG YAML")
ap.add_argument("--FF", nargs=8, type=float, default=[0, 0, 0, 1e3, 0, 0, 0, 0],
                metavar=tuple(FLBL),
                help="plate force resultants (default: M11 = 1e3 N)")
ap.add_argument("--npts", type=int, default=9, help="sample points per ply")
ap.add_argument("--out", default=None, help="also write the profile to this .dat")
a = ap.parse_args()
FF = np.array(a.FF, float)

# ------------------------------------------ read the 1-D SG: layup + mesh + materials
sg = read_plate_sg_yaml(a.yaml)
h = float(sum(sg["thick"]))
print("1-D SG : %s" % os.path.relpath(a.yaml, CC))
print("plies  : %s   (h = %.4f m, reference fraction = %.2f)"
      % (", ".join("%s(%.1fmm/%g)" % (m, 1e3 * t, x)
                   for m, t, x in zip(sg["mat_names"], sg["thick"], sg["angles"])),
         h, sg["fraction"]))

# ------------------------------------------------------------- RM homogenization
r = rm_plate_msg(sg["thick"], sg["angles"], sg["mat_names"], sg["material_db"],
                 n_per_layer=sg["n_per_layer"], elem_order=sg["elem_order"],
                 fraction=sg["fraction"])
print("\nHOMOGENIZATION -- RM 8x8 ABDG [[A,B,0],[B,D,0],[0,0,G]]"
      " (rows 1-6: e11,e22,g12,k11,k22,k12; rows 7-8: 2g13,2g23):")
print(r["ABDG"])
print("  Ustar_rel = %.2e  (unabsorbed 2nd-order energy)" % r["Ustar_rel"])

# --------------------------------------------- FF -> global plate strains + gradients
S6 = np.linalg.inv(r["A6"])
E6 = S6 @ FF[:6]                                  # membrane + bending strains
gam = np.linalg.solve(r["G_msg"], FF[6:])         # transverse shear [2g13, 2g23]
# equilibrium moment gradients (N, Q constant):  M11,1 = Q1 ,  M22,2 = Q2
dE1 = S6 @ np.array([0.0, 0.0, 0.0, FF[6], 0.0, 0.0])
dE2 = S6 @ np.array([0.0, 0.0, 0.0, 0.0, FF[7], 0.0])

print("\nDEHOMOGENIZATION -- applied plate resultants")
print("  FF    = %s" % np.array2string(FF, precision=4))
print("  E     = inv(A6) @ FF[:6]  = %s" % np.array2string(E6, precision=4))
print("  gamma = inv(G)  @ FF[6:]  = %s" % np.array2string(gam, precision=4))
print("  dE1   = inv(A6) @ [0 0 0 Q1 0 0] = %s   (equilibrium M11,1 = Q1)"
      % np.array2string(dE1, precision=4))
print("  dE2   = inv(A6) @ [0 0 0 0 Q2 0] = %s   (equilibrium M22,2 = Q2)"
      % np.array2string(dE2, precision=4))

# --------------------------------------------------------------- recovery per depth
bot = np.concatenate([[0.0], np.cumsum(sg["thick"])]) - sg["fraction"] * h
zs, plies = [], []
for k in range(len(sg["thick"])):
    za, zb = bot[k], bot[k + 1]
    zs.append(np.linspace(za + 1e-9 * (zb - za), zb - 1e-9 * (zb - za), a.npts))
    plies.append(np.full(a.npts, k))
zs = np.concatenate(zs); plies = np.concatenate(plies)

rows = []
for z in zs:
    Gam, Sig, ply = msgrm_strain_at_depth(r, z, E6, dE1, dE2)
    rows.append(np.concatenate([[z / h], np.asarray(Gam), np.asarray(Sig) / 1e6]))
rows = np.array(rows)

print("\n  through-thickness profile (stress in MPa, strain dimensionless)")
print("   %6s %3s | %s | %s" % ("z/h", "ply",
                                " ".join("%10s" % ("eps_" + s[1:]) for s in SLBL),
                                " ".join("%10s" % s for s in SLBL)))
for i in list(range(0, len(rows), max(1, a.npts // 3))) + [len(rows) - 1]:
    print("   %+6.3f %3d | %s | %s"
          % (rows[i, 0], plies[i],
             " ".join("%10.3e" % v for v in rows[i, 1:7]),
             " ".join("%10.3f" % v for v in rows[i, 7:])))

# ---------------------------- resultant check: re-integrate the recovered stress
z_m = zs
S_Pa = rows[:, 7:] * 1e6                          # (npts, 6) [s11 s22 s33 s23 s13 s12]
FF_re = np.array([np.trapezoid(S_Pa[:, 0], z_m),          # N11
                  np.trapezoid(S_Pa[:, 1], z_m),          # N22
                  np.trapezoid(S_Pa[:, 5], z_m),          # N12
                  np.trapezoid(S_Pa[:, 0] * z_m, z_m),    # M11
                  np.trapezoid(S_Pa[:, 1] * z_m, z_m),    # M22
                  np.trapezoid(S_Pa[:, 5] * z_m, z_m),    # M12
                  np.trapezoid(S_Pa[:, 4], z_m),          # Q1 = int s13
                  np.trapezoid(S_Pa[:, 3], z_m)])         # Q2 = int s23
print("\n  resultant check (int of the recovered stress through the thickness):")
print("   %6s %14s %14s" % ("", "applied", "re-integrated"))
for k in range(8):
    print("   %6s %14.4e %14.4e" % (FLBL[k], FF[k], FF_re[k]))

if a.out:
    hdr = ("MSG-RM FF-driven dehom | yaml %s\nFF   = %s\n"
           "E    = %s\ngamma= %s\ndE1  = %s\ndE2  = %s\n"
           "8x8 ABDG (rows e11,e22,g12,k11,k22,k12,2g13,2g23):\n%s\n"
           "Ustar_rel = %.6e\n\nprofile columns:\nz/h  %s  %s [MPa]"
           % (os.path.relpath(a.yaml, CC), np.array2string(FF),
              np.array2string(E6), np.array2string(gam),
              np.array2string(dE1), np.array2string(dE2),
              "\n".join("  " + " ".join("%14.6e" % v for v in row) for row in r["ABDG"]),
              r["Ustar_rel"],
              " ".join("eps_" + s[1:] for s in SLBL), " ".join(SLBL)))
    np.savetxt(a.out, rows, header=hdr, fmt="%14.6e")
    print("\n  wrote %s" % a.out)
