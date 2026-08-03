"""make_abaqus_ex2.py -- Nayak Example 2 (= the Khdeir-Reddy 1989 exact
benchmark, their ref [11]): the OpenSG-RM Abaqus decks.

PROBLEM STATEMENT:
    A simply supported (SS-1) SQUARE cross-ply laminate (0/90/0), equal-
    thickness plies, a = b = 5h with h = 0.1524 m -> a = 0.762 m (a THICK
    plate: a/h = 5, transverse shear dominates -- that is the point of the
    benchmark).  Material (the classic 25:1 graphite/epoxy set):
        E1 = 172.369 GPa, E2 = 6.895 GPa  (E1/E2 = 25)
        G12 = G13 = 3.448 GPa (= 0.5 E2), G23 = 1.379 GPa (= 0.2 E2)
        nu12 = 0.25, rho = 1603.03 kg/m^3
    Load: q(x,y,t) = q0 F(t) sin(pi x/a) sin(pi y/b), q0 = 68.9476 MPa;
    F(t) = STEP cut off at t1 = 6 ms (1 for t <= t1, 0 after) and the
    EXPLOSIVE BLAST e^(-330 t); window 20 ms.  Time step: dt = 50 us fixed
    (Nayak's accepted step; Khdeir-Reddy's solution is exact in time).
    Reported: center deflection w/0.0254 m and the top-surface normal
    stress sigma_x(a/2, b/2, h/2)/q0.

WHY THIS CASE COMES FIRST: Khdeir & Reddy (Composites Sci. & Tech. 34,
1989) give the EXACT ANALYTICAL transient solution of the higher-order
plate theory for this plate -- the analytic anchor of Nayak's whole paper
(his Tables 2-3 converge to it).  Comparing OpenSG-RM against that exact
solution isolates theory-vs-theory differences with NO discretization
noise on the reference side.

THE OpenSG-RM ROUTE (identical to Ex.5): the (0/90/0) laminate becomes a
through-thickness 1-D SG -> rm_plate_msg -> the MSG 8x8 ABDG -> a plain
Abaqus S4 model (*SHELL GENERAL SECTION + *TRANSVERSE SHEAR STIFFNESS)
-> *DYNAMIC with the double-sine load.  Prints: center-node U every
increment + SF/SM and COORD at the center 2x2-element patch (for the
sigma_x recovery at the top surface).

Run:  python examples/opensg-rm_dynamic/ex2/make_abaqus_ex2.py
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
# ALL CASE DATA (Nayak Ex.2 = Khdeir-Reddy 1989 digits)
# ----------------------------------------------------------------------------
H = 0.1524              # thickness [m]
A = 5.0 * H             # side a = b = 5h = 0.762 m (THICK plate)
NX = 20                 # S4 elements per side
Q0 = 68.9476e6          # load intensity [Pa]
T1 = 0.006              # step cut-off [s]
CBLAST = 330.0          # blast decay e^(-c t)
DT, TTOT = 5.0e-5, 0.02  # 50 us fixed increments, 20 ms window
MATERIAL_DB = {
    "ge25": {"E": [172.369e9, 6.895e9, 6.895e9],
             "G": [3.448e9, 3.448e9, 1.379e9],
             "nu": [0.25, 0.25, 0.25], "rho": 1603.03,
             "full_name": "graphite/epoxy, $E_1/E_2 = 25$"},
}
LAYUP = {"mat_names": ["ge25"] * 3, "thick": [H / 3.0] * 3,
         "angles": [0.0, 90.0, 0.0]}

yml = os.path.join(HERE, "ex2_sg.yaml")
plate_sg_yaml(yml, LAYUP, MATERIAL_DB, fraction=0.5)    # writes the yaml AND the png
inp = read_plate_sg_yaml(yml)
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
ABDG = np.asarray(r["ABDG"])
AB, G2 = ABDG[:6, :6], ABDG[6:8, 6:8]
rho_h = sum(MATERIAL_DB[m]["rho"] * t
            for m, t in zip(LAYUP["mat_names"], LAYUP["thick"]))
dx = A / NX


def n(i, j):
    return 1 + i + (NX + 1) * j


def e(i, j):
    return 1 + i + NX * j


def sinsin(i, j):
    """The double-sine load value at the element centroid."""
    xc, yc = (i + 0.5) * dx, (j + 0.5) * dx
    return np.sin(np.pi * xc / A) * np.sin(np.pi * yc / A)


def write_rm(kind):
    """One deck: kind = 'step' (cut off at T1) or 'blast' e^(-c t)."""
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]  # upper tri
    L = ["*HEADING",
         "Nayak Ex.2 / Khdeir-Reddy 1989 (0/90/0) a=5h, OpenSG-RM 8x8"
         " shell, %s pulse" % kind,
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
    L.append("*ELSET, ELSET=PATCHC")
    L.append(", ".join(str(v) for v in (e(c - 1, c - 1), e(c, c - 1),
                                        e(c - 1, c), e(c, c))))
    for name, ids in (("NX0", [n(0, j) for j in range(NX + 1)]),
                      ("NXA", [n(NX, j) for j in range(NX + 1)]),
                      ("NY0", [n(i, 0) for i in range(NX + 1)]),
                      ("NYB", [n(i, NX) for i in range(NX + 1)])):
        L.append("*NSET, NSET=%s" % name)
        for s in range(0, len(ids), 12):
            L.append(", ".join(str(v) for v in ids[s:s + 12]))
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append("1, %d, 1" % n(NX, NX))
    L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%.6g" % rho_h)
    for s in range(0, len(tri), 8):
        L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
    L.append("*TRANSVERSE SHEAR STIFFNESS")
    L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))
    L.append("** SS-1 + drilling")
    L.append("*BOUNDARY")
    L.append("NX0, 2, 3")
    L.append("NXA, 2, 3")
    L.append("NY0, 1, 1")
    L.append("NYB, 1, 1")
    L.append("NY0, 3, 3")
    L.append("NYB, 3, 3")
    L.append("NALL, 6, 6")
    L.append("*AMPLITUDE, NAME=FT")
    if kind == "step":
        # 1 up to t1 = 6 ms, then 0 (sharp drop over one increment)
        L.append("0., 1., %g, 1., %g, 0., %g, 0." % (T1, T1 + 1e-6, TTOT))
    else:
        ts = np.arange(0.0, TTOT + 1e-12, 2.5e-4)
        toks = []
        for t in ts:
            toks += ["%.6g" % t, "%.6g" % np.exp(-CBLAST * t)]
        for s in range(0, len(toks), 8):
            L.append(", ".join(toks[s:s + 8]))
    L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
    L.append("*DYNAMIC")
    L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
    L.append("*DLOAD, AMPLITUDE=FT")
    for j in range(NX):
        for i in range(NX):
            L.append("%d, P, %.6e" % (e(i, j), Q0 * sinsin(i, j)))
    L.append("*NODE PRINT, NSET=NCEN, FREQUENCY=1")
    L.append("U")
    L.append("*EL PRINT, ELSET=PATCHC, FREQUENCY=1")
    L.append("SF, SM")
    L.append("*EL PRINT, ELSET=PATCHC, FREQUENCY=1")
    L.append("COORD")
    L.append("*END STEP")
    path = os.path.join(HERE, "ex2_RM_%s.inp" % kind)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def amplitude_lines(kind):
    """The *AMPLITUDE FT block (step cut at T1, or tabulated blast)."""
    L = ["*AMPLITUDE, NAME=FT"]
    if kind == "step":
        L.append("0., 1., %g, 1., %g, 0., %g, 0." % (T1, T1 + 1e-6, TTOT))
    else:
        ts = np.arange(0.0, TTOT + 1e-12, 2.5e-4)
        toks = []
        for t in ts:
            toks += ["%.6g" % t, "%.6g" % np.exp(-CBLAST * t)]
        for s in range(0, len(toks), 8):
            L.append(", ".join(toks[s:s + 8]))
    return L


def write_fsdt(kind):
    """The CONVENTIONAL Abaqus 2-D shell route (FSDT): *SHELL SECTION,
    COMPOSITE with the three plies -- Abaqus builds A/B/D and ITS OWN
    transverse-shear stiffness (the standard industry first-order
    treatment).  Identical mesh/BCs/load/prints to the OpenSG-RM deck; the
    ONLY difference is who supplies the section law."""
    m = MATERIAL_DB["ge25"]
    L = ["*HEADING",
         "Nayak Ex.2 (0/90/0) a=5h, CONVENTIONAL Abaqus S4 composite"
         " section (FSDT), %s pulse" % kind,
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
    for name, ids in (("NX0", [n(0, j) for j in range(NX + 1)]),
                      ("NXA", [n(NX, j) for j in range(NX + 1)]),
                      ("NY0", [n(i, 0) for i in range(NX + 1)]),
                      ("NYB", [n(i, NX) for i in range(NX + 1)])):
        L.append("*NSET, NSET=%s" % name)
        for s in range(0, len(ids), 12):
            L.append(", ".join(str(v) for v in ids[s:s + 12]))
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append("1, %d, 1" % n(NX, NX))
    L.append("*MATERIAL, NAME=GE25")
    L.append("*ELASTIC, TYPE=LAMINA")
    L.append("%.6e, %.6e, %.4g, %.6e, %.6e, %.6e"
             % (m["E"][0], m["E"][1], m["nu"][0],
                m["G"][0], m["G"][1], m["G"][2]))
    L.append("*DENSITY")
    L.append("%g," % m["rho"])
    L.append("*SHELL SECTION, ELSET=EALL, COMPOSITE")
    for ang in (0.0, 90.0, 0.0):
        L.append("%.8g, 3, GE25, %g" % (H / 3.0, ang))
    L.append("** SS-1 + drilling")
    L.append("*BOUNDARY")
    for b in ("NX0, 2, 3", "NXA, 2, 3", "NY0, 1, 1", "NYB, 1, 1",
              "NY0, 3, 3", "NYB, 3, 3", "NALL, 6, 6"):
        L.append(b)
    L += amplitude_lines(kind)
    L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
    L.append("*DYNAMIC")
    L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
    L.append("*DLOAD, AMPLITUDE=FT")
    for j in range(NX):
        for i in range(NX):
            L.append("%d, P, %.6e" % (e(i, j), Q0 * sinsin(i, j)))
    L.append("*NODE PRINT, NSET=NCEN, FREQUENCY=1")
    L.append("U")
    L.append("*END STEP")
    path = os.path.join(HERE, "ex2_FSDT_%s.inp" % kind)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


NZP = 6                       # solid elements per ply (18 through h)


def write_solid(kind):
    """The 3-D BENCHMARK: 20x20x18 C3D8I (6 elements per ply), per-ply
    *ORIENTATION, same SS-1 supports and double-sine face pressure.  This
    is the arbiter the theories are judged against (Khdeir-Reddy's exact
    solution is exact only WITHIN the HSDT)."""
    m = MATERIAL_DB["ge25"]
    nzt = 3 * NZP
    NPL = (NX + 1) * (NX + 1)

    def n3(i, j, k):
        return 1 + i + (NX + 1) * j + NPL * k

    def e3(i, j, k):
        return 1 + i + NX * j + NX * NX * k

    L = ["*HEADING",
         "Nayak Ex.2 (0/90/0) a=5h, 3-D C3D8I benchmark, %s pulse" % kind,
         "*NODE"]
    dz = H / nzt
    for k in range(nzt + 1):
        for j in range(NX + 1):
            for i in range(NX + 1):
                L.append("%d, %.8f, %.8f, %.8f"
                         % (n3(i, j, k), i * dx, j * dx, k * dz))
    L.append("*ELEMENT, TYPE=C3D8I")
    for k in range(nzt):
        for j in range(NX):
            for i in range(NX):
                nn = [n3(i, j, k), n3(i + 1, j, k), n3(i + 1, j + 1, k),
                      n3(i, j + 1, k), n3(i, j, k + 1), n3(i + 1, j, k + 1),
                      n3(i + 1, j + 1, k + 1), n3(i, j + 1, k + 1)]
                L.append("%d, %s" % (e3(i, j, k),
                                     ", ".join(str(v) for v in nn)))
    for p in range(3):                       # ply elsets (6 layers each)
        L.append("*ELSET, ELSET=PLY%d, GENERATE" % (p + 1))
        L.append("%d, %d, 1" % (e3(0, 0, p * NZP),
                                e3(NX - 1, NX - 1, (p + 1) * NZP - 1)))
    c = NX // 2
    L.append("*NSET, NSET=NCEN3D")
    L.append("%d" % n3(c, c, nzt // 2))
    L.append("*ELSET, ELSET=COLC")
    ids = [str(e3(ii, jj, k)) for k in range(nzt)
           for (ii, jj) in ((c - 1, c - 1), (c, c - 1), (c - 1, c), (c, c))]
    for s in range(0, len(ids), 12):
        L.append(", ".join(ids[s:s + 12]))
    for name, sel in (("FX0", [(0, j) for j in range(NX + 1)]),
                      ("FXA", [(NX, j) for j in range(NX + 1)]),
                      ("FY0", [(i, 0) for i in range(NX + 1)]),
                      ("FYB", [(i, NX) for i in range(NX + 1)])):
        L.append("*NSET, NSET=%s" % name)
        ids = [str(n3(i, j, k)) for k in range(nzt + 1) for (i, j) in sel]
        for s in range(0, len(ids), 12):
            L.append(", ".join(ids[s:s + 12]))
    L.append("*MATERIAL, NAME=GE25")
    L.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    L.append("%.6e, %.6e, %.6e, %.4g, %.4g, %.4g, %.6e, %.6e,"
             % (m["E"][0], m["E"][1], m["E"][2], m["nu"][0], m["nu"][1],
                m["nu"][2], m["G"][0], m["G"][1]))
    L.append("%.6e" % m["G"][2])
    L.append("*DENSITY")
    L.append("%g," % m["rho"])
    for p, ang in enumerate((0.0, 90.0, 0.0)):
        L.append("*ORIENTATION, NAME=OR%d, SYSTEM=RECTANGULAR" % (p + 1))
        L.append("1.0, 0.0, 0.0, 0.0, 1.0, 0.0")
        L.append("3, %g" % ang)
        L.append("*SOLID SECTION, ELSET=PLY%d, MATERIAL=GE25,"
                 " ORIENTATION=OR%d" % (p + 1, p + 1))
    L.append("** SS-1 (same rule as the shells)")
    L.append("*BOUNDARY")
    for b in ("FX0, 2, 3", "FXA, 2, 3", "FY0, 1, 1", "FYB, 1, 1",
              "FY0, 3, 3", "FYB, 3, 3"):
        L.append(b)
    L += amplitude_lines(kind)
    L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
    L.append("*DYNAMIC")
    L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
    L.append("*DLOAD, AMPLITUDE=FT")
    for j in range(NX):
        for i in range(NX):
            L.append("%d, P2, %.6e" % (e3(i, j, nzt - 1), Q0 * sinsin(i, j)))
    L.append("*NODE PRINT, NSET=NCEN3D, FREQUENCY=1")
    L.append("U")
    L.append("*EL PRINT, ELSET=COLC, POSITION=CENTROIDAL, FREQUENCY=10")
    L.append("S")
    L.append("*END STEP")
    path = os.path.join(HERE, "ex2_SOLID_%s.inp" % kind)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")
print("OpenSG-RM 8x8 ABDG of the (0/90/0) a=5h laminate:")
for name, row in zip(ROWS, ABDG):
    print("%5s " % name + " ".join("%12.4e" % v for v in row))
print("rho*h = %.4f kg/m^2" % rho_h)
for kind in ("step", "blast"):
    print("wrote %s" % os.path.basename(write_rm(kind)))
    print("wrote %s" % os.path.basename(write_fsdt(kind)))
    print("wrote %s" % os.path.basename(write_solid(kind)))
