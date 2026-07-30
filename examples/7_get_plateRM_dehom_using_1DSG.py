"""Example 7 - MSG-RM plate HOMOGENIZATION + DEHOMOGENIZATION from a 1-D shell YAML,
driven by a plate-strain vector given on the command line.

Homogenization : core rm_plate_msg on the chosen wall laminate -> the 8x8 RM plate law
                 ABDG = [[A,B,0],[B,D,0],[0,0,G]]  (Yu 2003 Eqs. 40 and 61).
Dehomogenization: core msgrm_strain_at_depth -> the 3-D strain and stress through the
                 wall for the applied plate strain, Eq. (63) with the in-plane gradients
                 or Eq. (66) when the second gradients are also supplied.

The load is a UNIT PLATE STRAIN by default (unit kappa_11), passed as a 6-vector:

    E    = [e11, e22, g12, k11, k22, k12]      the plate strains themselves
    E,1  = dE/dx1  (same 6 components)         switches on the transverse shear s13/s23
    E,2  = dE/dx2  (same 6 components)
    E,11 / E,12 / E,22                         switch on the V2 sigma33 contribution

Run (defaults: st15 YAML, first laminate, unit k11):
    python examples/7_get_plateRM_dehom_using_1DSG.py
    python examples/7_get_plateRM_dehom_using_1DSG.py --section layup_2 --E 1e-3 0 0 0 0 0
    python examples/7_get_plateRM_dehom_using_1DSG.py --E 0 0 0 1 0 0 --dE1 0 0 0 0.5 0 0
    python examples/7_get_plateRM_dehom_using_1DSG.py --list          # show the laminates
    python examples/7_get_plateRM_dehom_using_1DSG.py --yaml my.yaml --out results.dat
"""
import argparse
import os
import sys

import numpy as np

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
np.set_printoptions(precision=4, linewidth=150)

from opensg_jax.fe_jax.msg_mesh import load_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from opensg_jax.fe_jax.msg_transverse_shear import plate_8x8

LBL = ["e11", "e22", "g12", "k11", "k22", "k12"]
SLBL = ["s11", "s22", "s33", "s23", "s13", "s12"]
DEFAULT_YAML = os.path.join(CC, "examples", "data", "1d_yaml", "st15_shell.yaml")

# --------------------------------------------------------------------- the arguments
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--yaml", default=DEFAULT_YAML, help="1-D shell SG YAML")
ap.add_argument("--section", default=None, help="elementSet name (default: the first)")
ap.add_argument("--list", action="store_true", help="list the laminates and exit")
ap.add_argument("--fraction", type=float, default=0.5,
                help="reference plane: 0 = OML, 0.5 = center (default), 1 = IML")
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

# ------------------------------------------------------------- read the 1-D SG YAML
_, _, mdb, layup_db, elem_to_layup = load_yaml(a.yaml)
names = list(layup_db)
if a.list:
    print("laminates in %s:" % os.path.relpath(a.yaml, CC))
    for nm in names:
        lay = layup_db[nm]
        n_el = sum(1 for v in elem_to_layup.values() if v == nm)
        print("  %-12s %d plies, h = %.4f m, %d elements : %s"
              % (nm, len(lay["thick"]), sum(float(t) for t in lay["thick"]), n_el,
                 ", ".join("%s(%.2fmm/%g)" % (m, 1e3 * float(t), float(x))
                           for m, t, x in zip(lay["mat_names"], lay["thick"],
                                              lay["angles"]))))
    sys.exit(0)

name = a.section or names[0]
if name not in layup_db:
    sys.exit("no laminate %r; available: %s" % (name, ", ".join(names)))
lay = layup_db[name]
thk = [float(t) for t in lay["thick"]]
ang = [float(x) for x in lay["angles"]]
mats = [str(m) for m in lay["mat_names"]]
h = float(sum(thk))

E6 = np.array(a.E, float)
grads = {k: (np.zeros(6) if getattr(a, k) is None else np.array(getattr(a, k), float))
         for k in ("dE1", "dE2", "dE11", "dE12", "dE22")}
second = any(np.any(grads[k]) for k in ("dE11", "dE12", "dE22"))

# ------------------------------------------------------------------- HOMOGENIZATION
r = rm_plate_msg(thk, ang, mats, mdb, fraction=a.fraction)
if r["G_msg"] is None:
    sys.exit("%s: fitted compliance not SPD (degenerate material?)" % name)
P8 = plate_8x8(np.asarray(r["A6"]), np.asarray(r["G_msg"]))

print("1-D SG      : %s" % os.path.relpath(a.yaml, CC))
print("laminate    : %s  (%d plies, h = %.4f m, reference fraction = %.2f)"
      % (name, len(thk), h, a.fraction))
print("plies       : %s" % ", ".join("%s(%.2fmm/%g)" % (m, 1e3 * t, x)
                                     for m, t, x in zip(mats, thk, ang)))
print("\nHOMOGENIZATION -- RM 8x8 ABDG [[A,B,0],[B,D,0],[0,0,G]]")
print("  rows/cols 1-6: %s ; 7-8: 2g13, 2g23" % ", ".join(LBL))
print(P8)
print("  G_msg diag = [%.4e %.4e]   Ustar_rel = %.2e  (unabsorbed 2nd-order energy)"
      % (r["G_msg"][0, 0], r["G_msg"][1, 1], r["Ustar_rel"]))

# ----------------------------------------------------------------- DEHOMOGENIZATION
print("\nDEHOMOGENIZATION -- applied plate strain")
print("  E     = %s" % np.array2string(E6, precision=4))
for k in ("dE1", "dE2", "dE11", "dE12", "dE22"):
    if np.any(grads[k]):
        print("  %-5s = %s" % (k, np.array2string(grads[k], precision=4)))
print("  order : %s (Eq. %s)" % (("SECOND, V2 active", "66") if second
                                 else ("first", "63")))
N = np.asarray(P8[:6, :6]) @ E6
print("  plate resultants A6 @ E = %s" % np.array2string(N, precision=4))

# sample points: --npts per ply, mid-referenced by the same fraction as the homo
bot = np.concatenate([[0.0], np.cumsum(thk)]) - a.fraction * h
zs, plies = [], []
for k in range(len(thk)):
    za, zb = bot[k], bot[k + 1]
    zz = np.linspace(za + 1e-9 * (zb - za), zb - 1e-9 * (zb - za), a.npts)
    zs.append(zz); plies.append(np.full(a.npts, k))
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
for i in range(0, len(rows), max(1, a.npts // 3)):
    print("   %+6.3f %3d | %s | %s"
          % (rows[i, 0], plies[i],
             " ".join("%10.3e" % v for v in rows[i, 1:7]),
             " ".join("%10.3f" % v for v in rows[i, 7:])))
print("   %+6.3f %3d | %s | %s"
      % (rows[-1, 0], plies[-1],
         " ".join("%10.3e" % v for v in rows[-1, 1:7]),
         " ".join("%10.3f" % v for v in rows[-1, 7:])))

imax = int(np.argmax(np.abs(rows[:, 7])))
print("\n  peak |s11| = %.3f MPa at z/h = %+.3f (ply %d);  faces s13 = %.3e, %.3e MPa"
      % (abs(rows[imax, 7]), rows[imax, 0], plies[imax], rows[0, 11], rows[-1, 11]))

if a.out:
    hdr = ("MSG-RM homo+dehom | yaml %s | laminate %s | fraction %.2f\n"
           "E    = %s\ndE1  = %s\ndE2  = %s\ndE11 = %s\ndE12 = %s\ndE22 = %s\n"
           "8x8 ABDG (rows e11,e22,g12,k11,k22,k12,2g13,2g23):\n%s\n"
           "Ustar_rel = %.6e\n\nprofile columns:\n"
           "z/h  %s  %s [MPa]"
           % (os.path.relpath(a.yaml, CC), name, a.fraction,
              np.array2string(E6), np.array2string(grads["dE1"]),
              np.array2string(grads["dE2"]), np.array2string(grads["dE11"]),
              np.array2string(grads["dE12"]), np.array2string(grads["dE22"]),
              "\n".join("  " + " ".join("%14.6e" % v for v in row) for row in P8),
              r["Ustar_rel"],
              " ".join("eps_" + s[1:] for s in SLBL), " ".join(SLBL)))
    np.savetxt(a.out, rows, header=hdr, fmt="%14.6e")
    print("\n  wrote %s" % a.out)
