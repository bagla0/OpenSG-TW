"""make_abaqus_dyn.py -- deck generator for the transient sandwich benchmark
(README.md: Nayak-class problem, Garg sandwich data, Yu-2003 dynamic-recovery
protocol).  Reads sandwich_sg.yaml (make_1dsg.py) and writes

    sandwich_RM.inp     20 x 20 S4 plate carrying the OpenSG-RM 8x8 as a
                        *SHELL GENERAL SECTION (+ the 2x2 shear, + the section
                        density), SS-1 edges, suddenly-applied uniform
                        pressure held constant, implicit dynamics dt = 5e-5 s
                        for 0.025 s.  Per-increment prints: U at the center
                        node; SF/SM and COORD at three 2x2 element patches
                        (center, x-edge middle, y-edge middle) -- the
                        recovery post-processor patch-fits these into the
                        resultant fields and their gradients per time step.
    sandwich_SOLID.inp  the benchmark: 20 x 20 x 12 C3D8I (2 elements per
                        face sheet, 8 through the core), per-layer material +
                        density, same BCs / load / time integration.
                        Prints: center mid-surface U every increment;
                        centroidal S along the matching element columns every
                        20th increment (1 ms cadence).

Variables
---------
n(i, j) / e(i, j)     shell node (21 per row) and element (20 per row) ids
n3(i, j, k)/e3(i,j,k) solid ids; k = thickness layer (z-planes non-uniform:
                      2 x 2.5 mm face, 8 x 5 mm core, 2 x 2.5 mm face)
NCEN / PATCHC ...     the center node and the three recovery patches
q0, DT, TTOT          10 kPa step pressure; 5e-5 s; 0.025 s
SS-1                  x-edges u3 = u2 = 0; y-edges u3 = u1 = 0 (same rule in
                      both models; drilling ur3 = 0 on the orthotropic shell)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from opensg_jax.fe_jax.segment_plate import read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

A = 0.5                      # plate side [m]
NX = 20                      # shell/solid in-plane elements per side
Q0 = 10.0e3                  # step pressure [Pa]
DT, TTOT = 5.0e-5, 0.025     # time step / total time [s]
NZF, NZC = 2, 8              # solid elements per face sheet / through the core

inp = read_plate_sg_yaml(os.path.join(HERE, "sandwich_sg.yaml"))
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
ABDG = np.asarray(r["ABDG"])
AB = ABDG[:6, :6]; G2 = ABDG[6:, 6:]
db = inp["material_db"]
thick = inp["thick"]; mats = inp["mat_names"]
H = float(sum(thick))
rho_h = sum(db[m]["rho"] * t for m, t in zip(mats, thick))
dx = A / NX


def n(i, j):
    return 1 + i + (NX + 1) * j


def e(i, j):
    return 1 + i + NX * j


def common_step(L, elements_load, load_kw):
    """The shared *AMPLITUDE + dynamic step block: suddenly applied constant
    pressure (step at t = 0, held), fixed-dt implicit dynamics."""
    L.append("*AMPLITUDE, NAME=STEPL")
    L.append("0., 1., %g, 1." % TTOT)
    L.append("**")
    L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
    L.append("*DYNAMIC")
    L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
    L.append("*DLOAD, AMPLITUDE=STEPL")
    for eid in elements_load:
        L.append("%d, %s, %.6e" % (eid, load_kw, Q0))


def write_rm(path):
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]
    L = []
    L.append("*HEADING")
    L.append("Transient sandwich plate, OpenSG-RM 8x8 shell (see README.md)")
    L.append("** a = b = %g m, h = %g m, [0/core/0] 0.1h/0.8h/0.1h" % (A, H))
    L.append("** step pressure q0 = %g Pa held; dt = %g s, T = %g s"
             % (Q0, DT, TTOT))
    L.append("*NODE")
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
    L.append("**")
    L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%.6g" % rho_h)
    for s in range(0, len(tri), 8):
        L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
    L.append("*TRANSVERSE SHEAR STIFFNESS")
    L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))
    L.append("**")
    L.append("** SS-1 simple supports + drilling (orthotropic layup)")
    L.append("*BOUNDARY")
    L.append("NX0, 2, 3")
    L.append("NXA, 2, 3")
    L.append("NY0, 1, 1")
    L.append("NYB, 1, 1")
    L.append("NY0, 3, 3")
    L.append("NYB, 3, 3")
    L.append("NALL, 6, 6")
    common_step(L, [e(i, j) for j in range(NX) for i in range(NX)], "P")
    L.append("*OUTPUT, FIELD, FREQUENCY=20")
    L.append("*ELEMENT OUTPUT")
    L.append("SF, SM")
    L.append("*NODE OUTPUT")
    L.append("U")
    L.append("*NODE PRINT, NSET=NCEN, FREQUENCY=1")
    L.append("U")
    for es in ("PATCHC", "PATCHX", "PATCHY"):
        L.append("*EL PRINT, ELSET=%s, FREQUENCY=1" % es)
        L.append("SF, SM")
        L.append("*EL PRINT, ELSET=%s, FREQUENCY=1" % es)
        L.append("COORD")
    L.append("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def write_solid(path):
    zk = np.concatenate([np.linspace(0, thick[0], NZF + 1)[:-1],
                         thick[0] + np.linspace(0, thick[1], NZC + 1)[:-1],
                         thick[0] + thick[1]
                         + np.linspace(0, thick[2], NZF + 1)])
    nzt = len(zk) - 1
    NPL = (NX + 1) * (NX + 1)

    def n3(i, j, k):
        return 1 + i + (NX + 1) * j + NPL * k

    def e3(i, j, k):
        return 1 + i + NX * j + NX * NX * k

    lay_of = lambda k: 0 if k < NZF else (1 if k < NZF + NZC else 2)
    L = []
    L.append("*HEADING")
    L.append("Transient sandwich plate, 3-D SOLID benchmark (see README.md)")
    L.append("** %d x %d x %d C3D8I; same BCs/load/time as sandwich_RM"
             % (NX, NX, nzt))
    L.append("*NODE")
    for k in range(nzt + 1):
        for j in range(NX + 1):
            for i in range(NX + 1):
                L.append("%d, %.8f, %.8f, %.8f"
                         % (n3(i, j, k), i * dx, j * dx, zk[k]))
    L.append("*ELEMENT, TYPE=C3D8I")
    for k in range(nzt):
        for j in range(NX):
            for i in range(NX):
                L.append("%d, %d, %d, %d, %d, %d, %d, %d, %d"
                         % (e3(i, j, k),
                            n3(i, j, k), n3(i + 1, j, k), n3(i + 1, j + 1, k),
                            n3(i, j + 1, k), n3(i, j, k + 1),
                            n3(i + 1, j, k + 1), n3(i + 1, j + 1, k + 1),
                            n3(i, j + 1, k + 1)))
    for m in range(3):
        L.append("*ELSET, ELSET=LAY%d" % (m + 1))
        ids = [str(e3(i, j, k)) for k in range(nzt) if lay_of(k) == m
               for j in range(NX) for i in range(NX)]
        for s in range(0, len(ids), 12):
            L.append(", ".join(ids[s:s + 12]))
    c = NX // 2
    L.append("*NSET, NSET=NCEN3D")
    L.append("%d" % n3(c, c, nzt // 2))
    for name, cols in (("COLC", [(c - 1, c - 1), (c, c - 1),
                                 (c - 1, c), (c, c)]),
                       ("COLX", [(0, c - 1), (0, c)]),
                       ("COLY", [(c - 1, 0), (c, 0)])):
        L.append("*ELSET, ELSET=%s" % name)
        ids = [str(e3(ii, jj, k)) for k in range(nzt) for (ii, jj) in cols]
        for s in range(0, len(ids), 12):
            L.append(", ".join(ids[s:s + 12]))
    for name, sel in (("NX0F", lambda i, j, k: i == 0),
                      ("NXAF", lambda i, j, k: i == NX),
                      ("NY0F", lambda i, j, k: j == 0),
                      ("NYBF", lambda i, j, k: j == NX)):
        L.append("*NSET, NSET=%s" % name)
        ids = [str(n3(i, j, k)) for k in range(nzt + 1)
               for j in range(NX + 1) for i in range(NX + 1)
               if sel(i, j, k)]
        for s in range(0, len(ids), 12):
            L.append(", ".join(ids[s:s + 12]))
    L.append("**")
    for name in ("face", "core"):
        m = db[name]
        L.append("*MATERIAL, NAME=%s" % name.upper())
        L.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
        L.append("%.6e, %.6e, %.6e, %.6g, %.6g, %.6g, %.6e, %.6e,"
                 % (m["E"][0], m["E"][1], m["E"][2], m["nu"][0], m["nu"][1],
                    m["nu"][2], m["G"][0], m["G"][1]))
        L.append("%.6e" % m["G"][2])
        L.append("*DENSITY")
        L.append("%g," % m["rho"])
    for m in range(3):
        L.append("*ORIENTATION, NAME=OR%d, SYSTEM=RECTANGULAR" % (m + 1))
        L.append("1.0, 0.0, 0.0, 0.0, 1.0, 0.0")
        L.append("3, %g" % inp["angles"][m])
        L.append("*SOLID SECTION, ELSET=LAY%d, MATERIAL=%s, ORIENTATION=OR%d"
                 % (m + 1, mats[m].upper(), m + 1))
    L.append("**")
    L.append("** SS-1 simple supports (same rule as the shell)")
    L.append("*BOUNDARY")
    L.append("NX0F, 2, 3")
    L.append("NXAF, 2, 3")
    L.append("NY0F, 1, 1")
    L.append("NYBF, 1, 1")
    L.append("NY0F, 3, 3")
    L.append("NYBF, 3, 3")
    common_step(L, [e3(i, j, nzt - 1) for j in range(NX) for i in range(NX)],
                "P2")
    L.append("*OUTPUT, FIELD, FREQUENCY=50")
    L.append("*ELEMENT OUTPUT")
    L.append("S")
    L.append("*NODE OUTPUT")
    L.append("U")
    L.append("*NODE PRINT, NSET=NCEN3D, FREQUENCY=1")
    L.append("U")
    for es in ("COLC", "COLX", "COLY"):
        L.append("*EL PRINT, ELSET=%s, POSITION=CENTROIDAL, FREQUENCY=20" % es)
        L.append("S")
    L.append("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    p1 = write_rm(os.path.join(HERE, "sandwich_RM.inp"))
    p2 = write_solid(os.path.join(HERE, "sandwich_SOLID.inp"))
    print("wrote %s and %s" % (os.path.basename(p1), os.path.basename(p2)))
    print("section mass rho*h = %.4f kg/m^2; D11 = %.4e; G11 = %.4e"
          % (rho_h, AB[3, 3], G2[0, 0]))
