"""Example 7 - MSG-RM plate homogenization + DEHOMOGENIZATION from a 1-D SG mesh YAML,
with the load given as a plate-strain vector on the command line.

Reads the same through-thickness 1-D SG as example 6, collects the layup information,
homogenizes with the core RM code (the 8x8 ABDG), then recovers the 3-D strain and
stress through the wall for the applied plate strain: Eq. (63) with the in-plane
gradients, or Eq. (66) when the second gradients are also supplied.

The load is a UNIT PLATE STRAIN by default (unit kappa_11), passed as 6-vectors:

    --E    = [e11, e22, g12, k11, k22, k12]    the plate strains themselves
    --dE1  = dE/dx1  (same 6 components)       switches on the transverse shear s13/s23
    --dE2  = dE/dx2
    --dE11 / --dE12 / --dE22                   switch on the V2 sigma33 contribution

Run:
    python examples/7_get_plateRM_dehom_using_1DSG.py
    python examples/7_get_plateRM_dehom_using_1DSG.py --E 1e-3 0 0 0 0 0
    python examples/7_get_plateRM_dehom_using_1DSG.py --E 0 0 0 1 0 0 --dE1 0 0 0 20 0 0
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

SLBL = ["s11", "s22", "s33", "s23", "s13", "s12"]
DEFAULT_YAML = os.path.join(CC, "examples", "data", "1d_yaml", "plate_sym45_sg.yaml")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--yaml", default=DEFAULT_YAML, help="through-thickness plate 1-D SG YAML")
ap.add_argument("--E", nargs=6, type=float, default=[0, 0, 0, 1, 0, 0],
                metavar=("e11", "e22", "g12", "k11", "k22", "k12"),
                help="plate strain vector (default: unit k11)")
ap.add_argument("--dE1", nargs=6, type=float, default=None, help="dE/dx1 (6 values)")
ap.add_argument("--dE2", nargs=6, type=float, default=None, help="dE/dx2 (6 values)")
ap.add_argument("--dE11", nargs=6, type=float, default=None, help="d2E/dx1dx1")
ap.add_argument("--dE12", nargs=6, type=float, default=None, help="d2E/dx1dx2")
ap.add_argument("--dE22", nargs=6, type=float, default=None, help="d2E/dx2dx2")
ap.add_argument("--npts", type=int, default=9, help="sample points per ply")
ap.add_argument("--out", default=None, help="also write the profile to this .dat")
a = ap.parse_args()

E6 = np.array(a.E, float)
grads = {k: (np.zeros(6) if getattr(a, k) is None else np.array(getattr(a, k), float))
         for k in ("dE1", "dE2", "dE11", "dE12", "dE22")}
second = any(np.any(grads[k]) for k in ("dE11", "dE12", "dE22"))

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

# ----------------------------------------------------------------- DEHOMOGENIZATION
print("\nDEHOMOGENIZATION -- applied plate strain")
print("  E     = %s" % np.array2string(E6, precision=4))
for k in ("dE1", "dE2", "dE11", "dE12", "dE22"):
    if np.any(grads[k]):
        print("  %-5s = %s" % (k, np.array2string(grads[k], precision=4)))
print("  order : %s (Eq. %s)" % (("SECOND, V2 active", "66") if second
                                 else ("first", "63")))
print("  plate resultants ABDG[:6,:6] @ E = %s"
      % np.array2string(r["ABDG"][:6, :6] @ E6, precision=4))

# sample points: --npts per ply, measured from the same reference plane as the homo
bot = np.concatenate([[0.0], np.cumsum(sg["thick"])]) - sg["fraction"] * h
zs, plies = [], []
for k in range(len(sg["thick"])):
    za, zb = bot[k], bot[k + 1]
    zs.append(np.linspace(za + 1e-9 * (zb - za), zb - 1e-9 * (zb - za), a.npts))
    plies.append(np.full(a.npts, k))
zs = np.concatenate(zs); plies = np.concatenate(plies)

rows = []
for z in zs:
    Gam, Sig, ply = msgrm_strain_at_depth(r, z, E6, grads["dE1"], grads["dE2"],
                                          dE11=grads["dE11"], dE12=grads["dE12"],
                                          dE22=grads["dE22"])
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

imax = int(np.argmax(np.abs(rows[:, 7])))
print("\n  peak |s11| = %.3f MPa at z/h = %+.3f (ply %d);  faces s13 = %.3e, %.3e MPa"
      % (abs(rows[imax, 7]), rows[imax, 0], plies[imax], rows[0, 11], rows[-1, 11]))

if a.out:
    hdr = ("MSG-RM homo+dehom | yaml %s\n"
           "E    = %s\ndE1  = %s\ndE2  = %s\ndE11 = %s\ndE12 = %s\ndE22 = %s\n"
           "8x8 ABDG (rows e11,e22,g12,k11,k22,k12,2g13,2g23):\n%s\n"
           "Ustar_rel = %.6e\n\nprofile columns:\n"
           "z/h  %s  %s [MPa]"
           % (os.path.relpath(a.yaml, CC),
              np.array2string(E6), np.array2string(grads["dE1"]),
              np.array2string(grads["dE2"]), np.array2string(grads["dE11"]),
              np.array2string(grads["dE12"]), np.array2string(grads["dE22"]),
              "\n".join("  " + " ".join("%14.6e" % v for v in row) for row in r["ABDG"]),
              r["Ustar_rel"],
              " ".join("eps_" + s[1:] for s in SLBL), " ".join(SLBL)))
    np.savetxt(a.out, rows, header=hdr, fmt="%14.6e")
    print("\n  wrote %s" % a.out)
