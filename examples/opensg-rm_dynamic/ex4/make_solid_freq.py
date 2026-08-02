"""make_solid_freq.py -- Nayak Ex.4: the 3-D SOLID frequency decks for the
mode-shape comparison against the OpenSG-RM shells.

Per layup: 24 x 12 x 9-layer C3D8I mesh of the cantilever plate (one
element per physical layer: 4 GE plies of 0.13 mm / 1 mm Al sheet / 4 GE
plies), per-ply *ORIENTATION, ENCASTRE on the x = 0 face, *FREQUENCY with
8 modes.  The RM mode shapes come from the EXISTING ex4_<slug>_freq.odb
shell runs; odb_modes.py renders both sets.

Run:  python examples/opensg-rm_dynamic/ex4/make_solid_freq.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_abaqus_freq import (AX, BY, TPLY, TAL, MATERIAL_DB, LAYUPS,
                              SLUGS)

NEX, NEY = 24, 12


def write_solid_freq(name, angles, mats, thick):
    slug = SLUGS[name]
    nzt = len(thick)
    dx, dy = AX / NEX, BY / NEY
    zk = [0.0]
    for t in thick:
        zk.append(zk[-1] + t)
    npl = (NEX + 1) * (NEY + 1)

    def n3(i, j, k):
        return 1 + i + (NEX + 1) * j + npl * k

    def e3(i, j, k):
        return 1 + i + NEX * j + NEX * NEY * k

    L = ["*HEADING",
         "Nayak Ex.4 cantilever %s -- 3-D C3D8I frequency benchmark"
         % slug,
         "*NODE"]
    for k in range(nzt + 1):
        for j in range(NEY + 1):
            for i in range(NEX + 1):
                L.append("%d, %.8f, %.8f, %.8f"
                         % (n3(i, j, k), i * dx, j * dy, zk[k]))
    L.append("*ELEMENT, TYPE=C3D8I")
    for k in range(nzt):
        for j in range(NEY):
            for i in range(NEX):
                nn = [n3(i, j, k), n3(i + 1, j, k), n3(i + 1, j + 1, k),
                      n3(i, j + 1, k), n3(i, j, k + 1),
                      n3(i + 1, j, k + 1), n3(i + 1, j + 1, k + 1),
                      n3(i, j + 1, k + 1)]
                L.append("%d, %s" % (e3(i, j, k),
                                     ", ".join(str(v) for v in nn)))
    for k in range(nzt):
        L.append("*ELSET, ELSET=LAY%d, GENERATE" % (k + 1))
        L.append("%d, %d, 1" % (e3(0, 0, k), e3(NEX - 1, NEY - 1, k)))
    L.append("*NSET, NSET=NROOT")
    ids = [str(n3(0, j, k)) for k in range(nzt + 1)
           for j in range(NEY + 1)]
    for s in range(0, len(ids), 12):
        L.append(", ".join(ids[s:s + 12]))
    done = set()
    for k, m in enumerate(mats):
        mu = m.upper()
        if mu in done:
            continue
        done.add(mu)
        md = MATERIAL_DB[m]
        L.append("*MATERIAL, NAME=%s" % mu)
        L.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
        L.append("%.6e, %.6e, %.6e, %.4g, %.4g, %.4g, %.6e, %.6e,"
                 % (md["E"][0], md["E"][1], md["E"][2], md["nu"][0],
                    md["nu"][1], md["nu"][2], md["G"][0], md["G"][1]))
        L.append("%.6e" % md["G"][2])
        L.append("*DENSITY")
        L.append("%g," % md["rho"])
    for k, (m, ang) in enumerate(zip(mats, angles)):
        L.append("*ORIENTATION, NAME=OR%d, SYSTEM=RECTANGULAR" % (k + 1))
        L.append("1.0, 0.0, 0.0, 0.0, 1.0, 0.0")
        L.append("3, %g" % ang)
        L.append("*SOLID SECTION, ELSET=LAY%d, MATERIAL=%s,"
                 " ORIENTATION=OR%d" % (k + 1, m.upper(), k + 1))
    L.append("*BOUNDARY")
    L.append("NROOT, 1, 3")
    L.append("*STEP, NAME=MODES")
    L.append("*FREQUENCY")
    L.append("8,")
    L.append("*END STEP")
    path = os.path.join(HERE, "ex4_%s_solidfreq.inp" % slug)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s (%d elements)" % (os.path.basename(path),
                                      NEX * NEY * nzt))


if __name__ == "__main__":
    for name, (angles, mats, thick) in LAYUPS.items():
        write_solid_freq(name, angles, mats, thick)
