"""make_fig5_rm.py -- Nayak Example 1 for OpenSG-RM: the Fig.-5 benchmark
(isotropic plate, suddenly applied CENTER PATCH load) run through the
OpenSG-RM route, to produce the same nondimensional deflection history.

PROBLEM STATEMENT (their Ex.1, exact solution by Reisman & Lee):
    Simply supported ISOTROPIC rectangular plate, a = 1.414, b = 1.0,
    h = 0.2 (thick: b/h = 5), E = 1.0, nu = 0.30, rho = 1.0 (consistent
    nondimensional units).  At t = 0 a uniform pressure q = 1.0 is
    suddenly applied on a SQUARE PATCH of side 0.4a centered on the plate
    and held (step in time, patch in space).  Reported: nondimensional
    center deflection  w_bar = w E a h / (q b^3)  versus nondimensional
    time  t_bar = (t/b) sqrt(E/rho)  (with these units t_bar = t).

THE OpenSG-RM ROUTE: the single isotropic layer becomes a 1-D SG ->
MSG 8x8 (its shear block reproduces the exact 5/6-type section shear with
no correction factor) -> plain S4 general-section deck -> *DYNAMIC with
fixed dt = 0.045 (200 steps over t_bar = 0 .. 9, finer than Nayak's own
dt = 0.10).  The patch edges fall ON element lines in BOTH directions:
x uniform (20 elements, patch = elements 7..14 exactly), y NON-UNIFORM
(6 + 8 + 6 intervals with lines at b/2 -+ 0.2a), so the applied load is
exact.  Boundary = HARD simple support (tangential edge rotations fixed),
the Navier/Reismann-Lee condition.

Run:  python examples/opensg-rm_dynamic/ex1/make_fig5_rm.py
then on the Abaqus machine:  abaqus job=ex1_RM_fig5 interactive
then:  python examples/opensg-rm_dynamic/ex1/plot_fig5.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg
from opensg_jax.fe_jax.segment_plate import plate_sg_yaml, \
    read_plate_sg_yaml

# ----------------------------------------------------------------------------
# ALL CASE DATA (Nayak Ex.1 digits, consistent nondimensional units)
# ----------------------------------------------------------------------------
AX, BY, H = 1.414, 1.0, 0.2
EMOD, NU, RHO = 1.0, 0.30, 1.0
QP = 1.0                    # patch pressure
PATCH = 0.4 * AX            # square patch side, centered
NEX, NEY = 20, 20
DT, TTOT = 0.045, 9.0
MATERIAL_DB = {"iso": {"E": [EMOD] * 3, "G": [EMOD / (2 * (1 + NU))] * 3,
                       "nu": [NU] * 3, "rho": RHO,
                       "full_name": "isotropic ($E=1$, $\\nu=0.3$)"}}

yml = os.path.join(HERE, "ex1_sg.yaml")
plate_sg_yaml(yml, {"mat_names": ["iso"], "thick": [H], "angles": [0.0]},
              MATERIAL_DB, fraction=0.5)
inp = read_plate_sg_yaml(yml)
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
ABDG = np.asarray(r["ABDG"])
AB, G2 = ABDG[:6, :6], ABDG[6:8, 6:8]
rho_h = RHO * H
dx = AX / NEX
# NON-UNIFORM y-lines so the patch edges fall ON element lines in y too
# (uniform y put the patch edge inside an element: 12 whole rows loaded =
# 0.60 m covered vs the true 0.5657 m -> +6 % total load).  6 + 8 + 6
# intervals: [0, y1], [y1, y2] (the patch), [y2, b].
y1, y2 = BY / 2 - PATCH / 2, BY / 2 + PATCH / 2
YL = np.concatenate([np.linspace(0.0, y1, 7)[:-1],
                     np.linspace(y1, y2, 9)[:-1],
                     np.linspace(y2, BY, 7)])


def n(i, j):
    return 1 + i + (NEX + 1) * j


def e(i, j):
    return 1 + i + NEX * j


tri = [AB[i, j] for j in range(6) for i in range(j + 1)]
L = ["*HEADING",
     "Nayak Ex.1 isotropic plate, center patch step load, OpenSG-RM S4",
     "*NODE"]
for j in range(NEY + 1):
    for i in range(NEX + 1):
        L.append("%d, %.8f, %.8f, 0.0" % (n(i, j), i * dx, YL[j]))
L.append("*ELEMENT, TYPE=S4, ELSET=EALL")
for j in range(NEY):
    for i in range(NEX):
        L.append("%d, %d, %d, %d, %d" % (e(i, j), n(i, j), n(i + 1, j),
                                         n(i + 1, j + 1), n(i, j + 1)))
c1, c2 = NEX // 2, NEY // 2
L.append("*NSET, NSET=NCEN")
L.append("%d" % n(c1, c2))
for name, ids in (("NX0", [n(0, j) for j in range(NEY + 1)]),
                  ("NXA", [n(NEX, j) for j in range(NEY + 1)]),
                  ("NY0", [n(i, 0) for i in range(NEX + 1)]),
                  ("NYB", [n(i, NEY) for i in range(NEX + 1)])):
    L.append("*NSET, NSET=%s" % name)
    for s in range(0, len(ids), 12):
        L.append(", ".join(str(v) for v in ids[s:s + 12]))
L.append("*NSET, NSET=NALL, GENERATE")
L.append("1, %d, 1" % n(NEX, NEY))
L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%.6g" % rho_h)
for s in range(0, len(tri), 8):
    L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
L.append("*TRANSVERSE SHEAR STIFFNESS")
L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))
L.append("** HARD simple support (the Navier / Reismann-Lee BC): w = 0")
L.append("** plus the TANGENTIAL rotation zero on every edge -- at x-edges")
L.append("** phi_y = UR1 (dof 4), at y-edges phi_x = UR2 (dof 5).  Soft SS")
L.append("** (rotations free) runs several % more flexible at h/b = 0.2.")
L.append("*BOUNDARY")
for b in ("NX0, 2, 4", "NXA, 2, 4", "NY0, 1, 1", "NYB, 1, 1",
          "NY0, 3, 3", "NYB, 3, 3", "NY0, 5, 5", "NYB, 5, 5",
          "NALL, 6, 6"):
    L.append(b)
L.append("*AMPLITUDE, NAME=FT")
L.append("0., 1., %g, 1." % TTOT)
L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
L.append("*DYNAMIC")
L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
L.append("*DLOAD, AMPLITUDE=FT")
npatch = 0
for j in range(NEY):
    for i in range(NEX):
        xc, yc = (i + 0.5) * dx, 0.5 * (YL[j] + YL[j + 1])
        if abs(xc - AX / 2) <= PATCH / 2 and abs(yc - BY / 2) <= PATCH / 2:
            L.append("%d, P, %.6e" % (e(i, j), QP))
            npatch += 1
L.append("*NODE PRINT, NSET=NCEN, FREQUENCY=1")
L.append("U")
L.append("*END STEP")
path = os.path.join(HERE, "ex1_RM_fig5.inp")
with open(path, "w") as f:
    f.write("\n".join(L) + "\n")
print("wrote %s (%d patch elements; D11=%.4e, G11=%.4e, rho_h=%.3g)"
      % (os.path.basename(path), npatch, AB[3, 3], G2[0, 0], rho_h))
