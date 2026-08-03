"""make_abaqus_ex3.py -- Nayak Example 3: STATIC bending of the classic
PAGANO sandwich plate -- the one case with an exact 3-D ELASTICITY anchor.

PROBLEM STATEMENT:
    Simply supported (SS-1) SQUARE sandwich plate (0/core/0) under the
    double-sine static pressure q(x,y) = q0 sin(pi x/a) sin(pi y/b).
    Faces (each 0.1h, 0 deg):  E1/E2 = 25, G12 = G13 = 0.5 E2,
        G23 = 0.2 E2, nu12 = 0.25, with E2 = 6.89 GPa (1e6 psi).
    Core (0.8h, Pagano's transversely stiff foam):
        E1 = E2 = 0.2756 GPa, G13 = G23 = 0.4134 GPa, G12 = 0.11024 GPa,
        nu = 0.25 (and E3 = 3.445 GPa = 0.5e6 psi from Pagano's original
        paper -- Nayak's 2-D theory does not use E3, the 1-D SG does).
    Two thickness ratios: a/h = 10 (thick) and a/h = 100 (thin).
    Exact reference: Pagano's 3-D elasticity solution (his ref [4]);
    literature FE: Nayak's 9-node element, 13 nodes per side on the
    quarter plate (his Table 4 finest column).

REPORTED (Nayak Table 4 normalizations):
    (sx, sy, sxy) x h^2/(q0 a^2)   at  sx, sy: (a/2, a/2, h/2);
                                       sxy: (0, 0, h/2)  [the corner]
    (sxz, syz)    x h/(q0 a)       at  sxz: (0, a/2, 0); syz: (a/2, 0, 0)
                                       [edge midspans, MID-PLANE z = 0]

THE OpenSG-RM ROUTE (same as ex2/ex5, but *STATIC): per a/h the layup
becomes a 1-D SG -> MSG 8x8 ABDG -> plain S4 general-section deck ->
one static solve -> recovery at the four stations from the patch SF/SM.

Run:  python examples/opensg-rm_dynamic/ex3/make_abaqus_ex3.py
then on the Abaqus machine:
  for %j in (ex3_RM_S10 ex3_RM_S100) do call abaqus job=%j interactive ask_delete=OFF
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
# ALL CASE DATA (Nayak Ex.3 = Pagano sandwich digits)
# ----------------------------------------------------------------------------
A = 1.0                 # side a = b [m] (results are nondimensional)
NX = 20                 # S4 elements per side
Q0 = 1.0e4              # load amplitude [Pa] (linear -- value cancels)
E2F = 6.89e9            # face transverse modulus (1e6 psi)
MATERIAL_DB = {
    "face": {"E": [25.0 * E2F, E2F, E2F],
             "G": [0.5 * E2F, 0.5 * E2F, 0.2 * E2F],
             "nu": [0.25, 0.25, 0.25], "rho": 1.0},
    "core": {"E": [0.2756e9, 0.2756e9, 3.445e9],
             "G": [0.11024e9, 0.4134e9, 0.4134e9],
             "nu": [0.25, 0.25, 0.25], "rho": 1.0},
}
SRATIOS = (10, 100)     # a/h


def n(i, j):
    return 1 + i + (NX + 1) * j


def e(i, j):
    return 1 + i + NX * j


def write_static(S):
    """One static deck for a/h = S: SG -> ABDG -> S4 -> *STATIC + the
    four recovery patches (center, corner, x-edge, y-edge)."""
    h = A / S
    layup = {"mat_names": ["face", "core", "face"],
             "thick": [0.1 * h, 0.8 * h, 0.1 * h],
             "angles": [0.0, 0.0, 0.0]}
    yml = os.path.join(HERE, "ex3_S%d_sg.yaml" % S)
    plate_sg_yaml(yml, layup, MATERIAL_DB, fraction=0.5)
    inp = read_plate_sg_yaml(yml)
    r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                     inp["material_db"], fraction=inp["fraction"])
    ABDG = np.asarray(r["ABDG"])
    AB, G2 = ABDG[:6, :6], ABDG[6:8, 6:8]
    dx = A / NX
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]
    L = ["*HEADING",
         "Nayak Ex.3 / Pagano sandwich (0/core/0), a/h=%d, OpenSG-RM S4"
         " STATIC" % S,
         "*NODE"]
    for j in range(NX + 1):
        for i in range(NX + 1):
            L.append("%d, %.8f, %.8f, 0.0" % (n(i, j), i * dx, j * dx))
    L.append("*ELEMENT, TYPE=S4, ELSET=EALL")
    for j in range(NX):
        for i in range(NX):
            L.append("%d, %d, %d, %d, %d" % (e(i, j), n(i, j), n(i + 1, j),
                                             n(i + 1, j + 1), n(i, j + 1)))
    c = NX // 2
    L.append("*NSET, NSET=NCEN")
    L.append("%d" % n(c, c))
    for name, ids in (("PATCHC", [e(c - 1, c - 1), e(c, c - 1),
                                  e(c - 1, c), e(c, c)]),
                      ("PATCHO", [e(0, 0), e(1, 0), e(0, 1), e(1, 1)]),
                      ("PATCHX", [e(0, c - 1), e(1, c - 1),
                                  e(0, c), e(1, c)]),
                      ("PATCHY", [e(c - 1, 0), e(c - 1, 1),
                                  e(c, 0), e(c, 1)])):
        L.append("*ELSET, ELSET=%s" % name)
        L.append(", ".join(str(v) for v in ids))
    for name, ids in (("NX0", [n(0, j) for j in range(NX + 1)]),
                      ("NXA", [n(NX, j) for j in range(NX + 1)]),
                      ("NY0", [n(i, 0) for i in range(NX + 1)]),
                      ("NYB", [n(i, NX) for i in range(NX + 1)])):
        L.append("*NSET, NSET=%s" % name)
        for s in range(0, len(ids), 12):
            L.append(", ".join(str(v) for v in ids[s:s + 12]))
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append("1, %d, 1" % n(NX, NX))
    L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=1.0")
    for s in range(0, len(tri), 8):
        L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
    L.append("*TRANSVERSE SHEAR STIFFNESS")
    L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))
    L.append("** SS-1 + drilling")
    L.append("*BOUNDARY")
    for b in ("NX0, 2, 3", "NXA, 2, 3", "NY0, 1, 1", "NYB, 1, 1",
              "NY0, 3, 3", "NYB, 3, 3", "NALL, 6, 6"):
        L.append(b)
    L.append("*STEP, NAME=BEND")
    L.append("*STATIC")
    L.append("1., 1.")
    L.append("*DLOAD")
    for j in range(NX):
        for i in range(NX):
            xc, yc = (i + 0.5) * dx, (j + 0.5) * dx
            L.append("%d, P, %.6e"
                     % (e(i, j), Q0 * np.sin(np.pi * xc / A)
                        * np.sin(np.pi * yc / A)))
    L.append("*NODE PRINT, NSET=NCEN")
    L.append("U")
    for es in ("PATCHC", "PATCHO", "PATCHX", "PATCHY"):
        L.append("*EL PRINT, ELSET=%s" % es)
        L.append("SF, SM")
        L.append("*EL PRINT, ELSET=%s" % es)
        L.append("COORD")
    L.append("*END STEP")
    path = os.path.join(HERE, "ex3_RM_S%d.inp" % S)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s  (h=%.4g, D11=%.4e, G11=%.4e)"
          % (os.path.basename(path), h, AB[3, 3], G2[0, 0]))


if __name__ == "__main__":
    for S in SRATIOS:
        write_static(S)
