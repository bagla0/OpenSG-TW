"""Example 7 - MSG-RM plate dehomogenization from a 1-D SG mesh YAML.

The user supplies ONE 8x1 vector -- either the global plate stress resultants or the
global plate strains (rows: e11,e22,g12,k11,k22,k12,2g13,2g23 / N11,N22,N12,M11,M22,
M12,Q1,Q2); the other is obtained through the homogenized 8x8 ABDG.  The through-
thickness recovery then uses the second-order warping function (Eq. 66 / Eq. 65) at the
GAUSS POINTS of the 1-D SG elements and writes three files, all in the LOCAL MATERIAL
FRAME of each ply:

    <base>.SM   6 stress components   [MPa]
    <base>.EM   6 strain components   [-]
    <base>.U    3 recovered local (warping) displacements   [m]

each row:  x1  x2  x3  <components>   (x3 measured from the SG reference plane).

Run:
    python examples/7_get_plateRM_dehom_using_1DSG.py --FF 0 0 0 1e3 0 0 1e3 0
    python examples/7_get_plateRM_dehom_using_1DSG.py --strain 0 0 0 0.96 -0.78 -0.1 8.5e-5 -4.3e-5
    python examples/7_get_plateRM_dehom_using_1DSG.py --yaml my_plate_sg.yaml --base out/my_case
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
from opensg_jax.fe_jax.msg_rm_plate import (rm_plate_msg, msgrm_strain_at_depth,
                                            msgrm_warping_at_depth)
from opensg_jax.fe_jax.msg_materials import rotation_6x6

FLBL = ["N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2"]
ELBL = ["e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23"]
DEFAULT_YAML = os.path.join(CC, "examples", "data", "1d_yaml", "plate_sym45_sg.yaml")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--yaml", default=DEFAULT_YAML, help="through-thickness plate 1-D SG YAML")
grp = ap.add_mutually_exclusive_group()
grp.add_argument("--FF", nargs=8, type=float, default=None, metavar=tuple(FLBL),
                 help="global plate stress resultants (default: M11 = 1e3, Q1 = 1e3)")
grp.add_argument("--strain", nargs=8, type=float, default=None, metavar=tuple(ELBL),
                 help="global plate strains instead of resultants")
ap.add_argument("--base", default=None,
                help="output basename for .SM/.EM/.U (default: alongside the yaml)")
a = ap.parse_args()

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

# ------------------------- the 8x1 input: resultants OR strains, closed by the 8x8
if a.strain is not None:
    EG8 = np.array(a.strain, float)
    FF = r["ABDG"] @ EG8
    given = "strain"
else:
    FF = np.array([0, 0, 0, 1e3, 0, 0, 1e3, 0], float) if a.FF is None \
        else np.array(a.FF, float)
    EG8 = np.linalg.solve(r["ABDG"], FF)
    given = "FF"
E6 = EG8[:6]
# strain-gradient drivers of the recovery, from plate equilibrium under constant N and Q
# (M11,1 = Q1, M22,2 = Q2); second gradients vanish for this state but the recovery is
# evaluated through the full second-order warping formula
S6 = np.linalg.inv(r["A6"])
dE1 = S6 @ np.array([0.0, 0.0, 0.0, FF[6], 0.0, 0.0])
dE2 = S6 @ np.array([0.0, 0.0, 0.0, 0.0, FF[7], 0.0])
zero6 = np.zeros(6)

print("\nDEHOMOGENIZATION -- computed using the second-order warping function")
print("  given %s = %s" % (given, np.array2string(FF if given == "FF" else EG8,
                                                  precision=4)))

# ------------------------------------------ recovery at the SG element GAUSS points
p = sg["elem_order"]
node_x = r["node_x"]
n_elem = len(sg["thick"]) * sg["n_per_layer"]
xi_g, _ = np.polynomial.legendre.leggauss(max(3, p + 1))

rows_S, rows_E, rows_U = [], [], []
for e in range(n_elem):
    za, zb = node_x[p * e], node_x[p * e + p]
    for xi in xi_g:
        z = 0.5 * (za + zb) + 0.5 * (zb - za) * xi
        Gam, Sig, th = msgrm_strain_at_depth(r, z, E6, dE1, dE2,
                                             dE11=zero6, dE12=zero6, dE22=zero6)
        w = msgrm_warping_at_depth(r, z, E6, dE1, dE2,
                                   dE11=zero6, dE12=zero6, dE22=zero6)
        # ---- rotate to the LOCAL MATERIAL FRAME of this ply (angle about x3) ----
        th = float(th)
        R = np.asarray(rotation_6x6(th))              # material -> global (stress Voigt)
        S_m = np.linalg.solve(R, np.asarray(Sig))     # sigma_mat = R^-1 sigma_glob
        E_m = R.T @ np.asarray(Gam)                   # eps_mat = R^T eps_glob (conjugate)
        c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        Q3 = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        U_m = Q3 @ np.asarray(w)
        rows_S.append([0.0, 0.0, z] + list(S_m / 1e6))
        rows_E.append([0.0, 0.0, z] + list(E_m))
        rows_U.append([0.0, 0.0, z] + list(U_m))

# ----------------------------------------------------------------- the three files
base = a.base or os.path.splitext(a.yaml)[0]
load_line = "%s = %s" % (given, np.array2string(FF if given == "FF" else EG8))
common = ("MSG-RM second-order dehomogenization | 1-D SG %s\n%s\n"
          "LOCAL MATERIAL FRAME (ply angle about x3) | rows at SG element Gauss points\n"
          % (os.path.relpath(a.yaml, CC), load_line))
for ext, rows, cols in (
        (".SM", rows_S, "x1 x2 x3  s11 s22 s33 s23 s13 s12  [MPa]"),
        (".EM", rows_E, "x1 x2 x3  e11 e22 e33 2e23 2e13 2e12  [-]"),
        (".U", rows_U, "x1 x2 x3  u1 u2 u3  [m]")):
    np.savetxt(base + ext, np.array(rows), header=common + cols, fmt="%15.6e")
    print("  wrote %s  (%d Gauss points)" % (os.path.relpath(base + ext, CC), len(rows)))
