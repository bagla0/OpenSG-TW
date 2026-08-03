"""abaqus_inp.py -- layup_db.yaml + 1dsg.yaml  ->  layup_db_abaqus.inp

The plate law is homogenized here from the SG that 1d_sg.py wrote, rather
than parsed back out of its .out -- one rm_plate_msg call is cheaper than
carrying a text parser and keeping it in step with the .out layout, and the
deck gets full precision instead of the printed six figures.

A plain Abaqus S4 plate whose ONLY constitutive input is the homogenized
plate law: the 6x6 goes in as *SHELL GENERAL SECTION (DENSITY = rho h) and,
when model = 1, the 2x2 MSG block as *TRANSVERSE SHEAR STIFFNESS.  No plies,
no orientations, no shear correction factor anywhere in the deck.

Simply supported (SS-1) on all four edges, drilling dof fixed (flat plate),
double-sine pressure q0 sin(pi x/a) sin(pi y/b) held over the step, implicit
*DYNAMIC at a fixed increment, centre deflection printed every increment.

Run:  python abaqus_inp.py
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from opensg_jax.fe_jax.segment_plate import read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

# ---- input -----------------------------------------------------------------
DB = os.path.join(HERE, "layup_db.yaml")
db = yaml.safe_load(open(DB))
model = int(db["model"])                 # 0 = ABD only, 1 = ABD + shear
fraction = float(db["fraction"])         # no code-side default: the YAML is
                                         # the ONLY place these are set
p = db["plate"]
A, B = float(p["a"]), float(p["b"])
NX, NY = int(p["nx"]), int(p["ny"])
Q0, DT, TTOT = float(p["q0"]), float(p["dt"]), float(p["ttot"])

# the section law: homogenize the SG that 1d_sg.py wrote.  n_per_layer and
# elem_order come back from the mesh file itself, so the section can never
# disagree with the mesh it was built on.
inp = read_plate_sg_yaml(os.path.join(HERE, "1dsg.yaml"))
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], n_per_layer=inp["n_per_layer"],
                 elem_order=inp["elem_order"], fraction=fraction)
AB = np.asarray(r["A6"])
G2 = np.asarray(r["ABDG"])[6:8, 6:8] if model == 1 else None
rho_h = sum(inp["material_db"][m]["rho"] * t
            for m, t in zip(inp["mat_names"], inp["thick"]))
print("model %d: %s -- the deck %s the *TRANSVERSE SHEAR STIFFNESS card"
      % (model, "shear-refined" if model == 1 else "classical",
         "carries" if model == 1 else "omits"))
print("section from 1dsg.yaml, reference fraction %g, rho*h = %.4f kg/m^2"
      % (fraction, rho_h))

# ---- mesh helpers ----------------------------------------------------------
dx, dy = A / NX, B / NY
n = lambda i, j: 1 + i + (NX + 1) * j            # node id, x fastest
e = lambda i, j: 1 + i + NX * j                  # element id

L = ["*HEADING",
     "OpenSG-RM plate from %s (model %d)" % (os.path.basename(DB), model),
     "*NODE"]
for j in range(NY + 1):
    for i in range(NX + 1):
        L.append("%d, %.8f, %.8f, 0.0" % (n(i, j), i * dx, j * dy))
L.append("*ELEMENT, TYPE=S4, ELSET=EALL")
for j in range(NY):
    for i in range(NX):
        L.append("%d, %d, %d, %d, %d" % (e(i, j), n(i, j), n(i + 1, j),
                                         n(i + 1, j + 1), n(i, j + 1)))
for nm, ids in (("NX0", [n(0, j) for j in range(NY + 1)]),
                ("NXA", [n(NX, j) for j in range(NY + 1)]),
                ("NY0", [n(i, 0) for i in range(NX + 1)]),
                ("NYB", [n(i, NY) for i in range(NX + 1)])):
    L.append("*NSET, NSET=%s" % nm)
    for s in range(0, len(ids), 12):
        L.append(", ".join(str(v) for v in ids[s:s + 12]))
L.append("*NSET, NSET=NALL, GENERATE")
L.append("1, %d, 1" % n(NX, NY))
L.append("*NSET, NSET=NCEN")
L.append("%d" % n(NX // 2, NY // 2))

# ---- the section: the 21 upper-triangle terms, COLUMN by column ------------
L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%.6g" % rho_h)
tri = [AB[i, j] for j in range(6) for i in range(j + 1)]
for s in range(0, len(tri), 8):
    L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
if model == 1:                                   # K11, K22, K12
    L.append("*TRANSVERSE SHEAR STIFFNESS")
    L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))

L.append("** SS-1: v=w=0 on the x-edges, u=w=0 on the y-edges, drilling fixed")
L.append("*BOUNDARY")
for card in ("NX0, 2, 3", "NXA, 2, 3", "NY0, 1, 1", "NYB, 1, 1",
             "NY0, 3, 3", "NYB, 3, 3", "NALL, 6, 6"):
    L.append(card)

L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
L.append("*DYNAMIC")
L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
L.append("*DLOAD")
for j in range(NY):                              # sine at the element centre
    for i in range(NX):
        q = Q0 * np.sin(np.pi * (i + 0.5) * dx / A) \
               * np.sin(np.pi * (j + 0.5) * dy / B)
        L.append("%d, P, %.6e" % (e(i, j), q))
L.append("*NODE PRINT, NSET=NCEN, FREQUENCY=1")
L.append("U")
L.append("*END STEP")

# ---- output ----------------------------------------------------------------
out = os.path.splitext(DB)[0] + "_abaqus.inp"
open(out, "w").write("\n".join(L) + "\n")
print("%s -> %s  (%d S4, rho_h = %.4f kg/m^2, %s)"
      % (os.path.basename(DB), os.path.basename(out), NX * NY, rho_h,
         "with *TRANSVERSE SHEAR STIFFNESS" if model == 1
         else "ABD only, no shear card"))
