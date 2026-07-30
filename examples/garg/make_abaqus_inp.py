"""make_abaqus_inp.py -- generate a complete, submit-ready Abaqus .inp for the Garg
cylindrical-bending strip, with the MSG-RM 8x8 ABDG installed as the shell section.

For each case (or --case), at aspect ratio --S (default 10):

  geometry   strip 0 <= x <= a (a = 1 m), one element wide (width = a/n), S4 shells,
             n = 100 elements along the span, reference surface = laminate mid-surface
  section    *SHELL GENERAL SECTION with the 21 constants of [[A,B],[B,D]] from the
             MSG-RM homogenization, + *TRANSVERSE SHEAR STIFFNESS with the MSG 2x2 G
             (K11 = G_13-13, K22 = G_23-23, K12 = coupling)
  load       q(x) = q0 sin(pi x / a) on the face, piecewise-constant per element
             column (element-centre value; with n = 100 the quadrature error ~0.04%)
  BCs        cylindrical bending: u2 = ur1 = ur3 = 0 on ALL nodes (nothing varies
             with x2); simple supports: u3 = 0 on both end edges, u1 = 0 at x = 0
  output     field SF/SM (section forces incl. SF4/SF5 = Q1/Q2) + U to the .odb;
             .dat prints of U at the mid-span node and SF/SM in the mid-span and
             end elements -- the FF extraction stations (M max at a/2, Q max at 0)

The header comments carry the closed-form MSG prediction |w| = q0/(p^4 D11) +
q0/(p^2 G11) and the exact 3-D amplitude, so one glance at the .dat U3 after the job
validates the run.  Submit manually:   abaqus job=garg_caseA_S10 interactive

Run:
    python examples/garg/make_abaqus_inp.py            # all three cases, S = 10
    python examples/garg/make_abaqus_inp.py --case caseC --S 4
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CC)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CC, "examples", "TW-paper", "rm_thickness"))

from exact_cyl import ExactCyl
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg
from garg_layups import MATERIAL_DB, LAYUPS, H

ap = argparse.ArgumentParser()
ap.add_argument("--case", default=None, help="caseA / caseB / caseC (default: all)")
ap.add_argument("--S", type=float, default=10.0, help="aspect ratio a/h (default 10)")
ap.add_argument("--nel", type=int, default=100, help="elements along the span")
a_ = ap.parse_args()

q0 = 1.0e4
a = 1.0
p = np.pi / a
NE = a_.nel
b = a / NE                                     # one square element across the width


def fmt(vals):
    """Abaqus data lines, at most 8 entries per line."""
    out = []
    for i in range(0, len(vals), 8):
        out.append(", ".join("%.6e" % v for v in vals[i:i + 8]))
    return "\n".join(out)


for name, lay0 in LAYUPS.items():
    if a_.case and name != a_.case:
        continue
    h = a / a_.S
    fr = [t / H for t in lay0["thick"]]
    thk = [f * h for f in fr]
    ang = lay0["angles"]; mats = lay0["mat_names"]

    r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
    AB = np.asarray(r["ABDG"][:6, :6])          # [[A,B],[B,D]]
    G2 = np.asarray(r["ABDG"][6:, 6:])
    D11 = float(AB[3, 3]); G11 = float(G2[0, 0])
    w_msg = q0 / (p ** 4 * D11) + q0 / (p ** 2 * G11)

    ex = ExactCyl(thk, ang, mats, MATERIAL_DB, a, q0=q0)
    zc, _, _, uvw = ex.profile(n_per_layer=41)
    w_ex = float(uvw[np.argmin(np.abs(zc)), 2])

    # 21 constants, Abaqus *SHELL GENERAL SECTION order: lower triangle by columns
    # D(1,1), D(1,2), D(2,2), D(1,3), D(2,3), D(3,3), D(1,4), ...
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]

    L = []
    A = L.append
    A("*HEADING")
    A("Garg cylindrical-bending strip, MSG-RM 8x8 general shell section")
    A("** case %s: plies %s" % (name, ", ".join(
        "%s(%.4gmm/%g)" % (m, 1e3 * t, x) for m, t, x in zip(mats, thk, ang))))
    A("** S = a/h = %g,  a = %g m,  h = %g m,  q0 = %g Pa" % (a_.S, a, h, q0))
    A("** load q(x) = q0*sin(pi*x/a), piecewise-constant per element column")
    A("** PREDICTED mid-span |w|: MSG closed form %.6e m ; exact 3-D %.6e m" % (w_msg, w_ex))
    A("**   (check the .dat U3 at node NMID against these after the job)")
    A("** D11 = %.6e ; G11 = %.6e ; G22 = %.6e" % (D11, G11, float(G2[1, 1])))
    A("**")
    A("*NODE")
    nid = lambda i, j: i * 2 + j + 1
    for i in range(NE + 1):
        for j in (0, 1):
            A("%d, %.8f, %.8f, 0.0" % (nid(i, j), i * a / NE, j * b))
    A("*ELEMENT, TYPE=S4, ELSET=EALL")
    for i in range(NE):
        A("%d, %d, %d, %d, %d" % (i + 1, nid(i, 0), nid(i + 1, 0),
                                  nid(i + 1, 1), nid(i, 1)))
    # per-column elsets only where needed; the load goes on element ids directly
    A("*ELSET, ELSET=EMID")
    A("%d" % (NE // 2))                          # element just left of x = a/2
    A("*ELSET, ELSET=EEND")
    A("1")
    A("*NSET, NSET=NX0")
    A("%d, %d" % (nid(0, 0), nid(0, 1)))
    A("*NSET, NSET=NXA")
    A("%d, %d" % (nid(NE, 0), nid(NE, 1)))
    A("*NSET, NSET=NMID")
    A("%d" % nid(NE // 2, 0))
    A("*NSET, NSET=NALL, GENERATE")
    A("1, %d, 1" % nid(NE, 1))
    A("**")
    A("** ---- the MSG-RM section: [[A,B],[B,D]] 21 constants + the 2x2 shear G ----")
    A("*SHELL GENERAL SECTION, ELSET=EALL")
    A(fmt(tri))
    A("*TRANSVERSE SHEAR STIFFNESS")
    A("%.6e, %.6e, %.6e" % (float(G2[0, 0]), float(G2[1, 1]), float(G2[0, 1])))
    A("**")
    A("** ---- cylindrical bending + simple supports ----")
    A("*BOUNDARY")
    A("NALL, 2, 2")                              # u2 = 0        (nothing varies with x2)
    A("NALL, 4, 4")                              # ur1 = 0       (kappa22 = 0)
    A("NALL, 6, 6")                              # ur3 = 0       (drilling)
    A("NX0, 3, 3")                               # w = 0 at x = 0
    A("NXA, 3, 3")                               # w = 0 at x = a
    A("NX0, 1, 1")                               # axial anchor
    A("**")
    A("*STEP, NAME=SINELOAD")
    A("*STATIC")
    A("** pressure on the shell face, element-centre value of q0*sin(pi*x/a)")
    A("*DLOAD")
    for i in range(NE):
        xc = (i + 0.5) * a / NE
        A("%d, P, %.6e" % (i + 1, q0 * np.sin(p * xc)))
    A("*OUTPUT, FIELD")
    A("*ELEMENT OUTPUT")
    A("SF, SM")
    A("*NODE OUTPUT")
    A("U")
    A("*NODE PRINT, NSET=NMID")
    A("U")
    A("*EL PRINT, ELSET=EMID")                   # M max here  -> FF for s11/s33 station
    A("SF, SM")
    A("*EL PRINT, ELSET=EEND")                   # Q max here  -> FF for s13 station
    A("SF, SM")
    A("*END STEP")

    out = os.path.join(HERE, name, "garg_%s_S%g.inp" % (name, a_.S))
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s  (D11 %.4e, G11 %.4e; predicted |w| MSG %.5e, exact %.5e)"
          % (os.path.relpath(out, CC), D11, G11, w_msg, w_ex))
