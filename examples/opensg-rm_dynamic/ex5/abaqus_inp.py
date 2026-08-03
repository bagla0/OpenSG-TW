"""abaqus_inp.py -- layup_db.yaml + layup_db_plate_homo.out  ->  layup_db.inp

The plate law is READ from the .out that 1d_sg.py already wrote, not
recomputed -- so this script needs neither JAX nor opensg_jax, just the
YAML for the plate and the .out for the section.

A plain Abaqus S4 plate whose ONLY constitutive input is the homogenized
plate law: the 6x6 goes in as *SHELL GENERAL SECTION (DENSITY = rho h) and,
when model = 1, the 2x2 MSG block as *TRANSVERSE SHEAR STIFFNESS.  No plies,
no orientations, no shear correction factor anywhere in the deck.

Simply supported (SS-1) on all four edges, drilling dof fixed (flat plate),
double-sine pressure q0 sin(pi x/a) sin(pi y/b) held over the step, implicit
*DYNAMIC at a fixed increment, centre deflection printed every increment.

Run:  python abaqus_inp.py [layup_db.yaml]
"""
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def read_homo(path):
    """The plate law and section mass out of the .out that 1d_sg.py wrote.

    The matrix is the block between the "rows/cols:" header and the blank
    line after it; the mass is on the "section mass" line.
    """
    M, mass, on = [], None, False
    for ln in open(path):
        if ln.startswith("rows/cols:"):
            on = True
        elif ln.startswith("section mass"):
            mass, on = float(ln.split("=")[1].split()[0]), False
        elif on:
            tok = ln.split()
            if tok:
                M.append([float(v) for v in tok])
            else:
                on = False
    return np.array(M), mass


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

# the section law, already computed by 1d_sg.py -- read it, do not redo it
M, rho_h = read_homo(os.path.splitext(DB)[0] + "_plate_homo.out")
AB = M[:6, :6]
G2 = M[6:8, 6:8] if model == 1 else None
print("model %d: %s -- the deck %s the *TRANSVERSE SHEAR STIFFNESS card"
      % (model, "shear-refined" if model == 1 else "classical",
         "carries" if model == 1 else "omits"))
print("section from %s, reference fraction %g, rho*h = %.4f kg/m^2"
      % (os.path.basename(os.path.splitext(DB)[0] + "_plate_homo.out"),
         fraction, rho_h))

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
out = os.path.splitext(DB)[0] + ".inp"
open(out, "w").write("\n".join(L) + "\n")
print("%s -> %s  (%d S4, rho_h = %.4f kg/m^2, %s)"
      % (os.path.basename(DB), os.path.basename(out), NX * NY, rho_h,
         "with *TRANSVERSE SHEAR STIFFNESS" if model == 1
         else "ABD only, no shear card"))
