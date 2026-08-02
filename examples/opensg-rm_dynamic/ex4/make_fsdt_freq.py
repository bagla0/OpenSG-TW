"""make_fsdt_freq.py -- Nayak Ex.4: the conventional Abaqus FSDT
(Whitney-1973-type first-order composite shell) frequency decks, for the
three-way mode comparison RM / FSDT / 3-D solid.

Per layup: the same 24 x 12 S4 cantilever mesh as make_abaqus_freq.py,
but the section is *SHELL SECTION, COMPOSITE listing the physical layers
(GE plies + the Al sheet as LAMINA materials) -- Abaqus builds A/B/D and
its own transverse shear.  *FREQUENCY, 8 modes.

Run:  python examples/opensg-rm_dynamic/ex4/make_fsdt_freq.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_abaqus_freq import (AX, BY, MATERIAL_DB, LAYUPS, SLUGS,
                              NEX, NEY)


def write_fsdt_freq(name, angles, mats, thick):
    slug = SLUGS[name]
    dx, dy = AX / NEX, BY / NEY

    def n(i, j):
        return 1 + i + (NEX + 1) * j

    L = ["*HEADING",
         "Nayak Ex.4 cantilever %s -- conventional Abaqus composite S4"
         " (FSDT) frequency" % slug,
         "*NODE"]
    for j in range(NEY + 1):
        for i in range(NEX + 1):
            L.append("%d, %.8f, %.8f, 0.0" % (n(i, j), i * dx, j * dy))
    L.append("*ELEMENT, TYPE=S4, ELSET=EALL")
    for j in range(NEY):
        for i in range(NEX):
            L.append("%d, %d, %d, %d, %d"
                     % (1 + i + NEX * j, n(i, j), n(i + 1, j),
                        n(i + 1, j + 1), n(i, j + 1)))
    L.append("*NSET, NSET=NROOT")
    ids = [str(n(0, j)) for j in range(NEY + 1)]
    for s in range(0, len(ids), 12):
        L.append(", ".join(ids[s:s + 12]))
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append("1, %d, 1" % n(NEX, NEY))
    done = set()
    for m in mats:
        mu = m.upper()
        if mu in done:
            continue
        done.add(mu)
        md = MATERIAL_DB[m]
        L.append("*MATERIAL, NAME=%s" % mu)
        L.append("*ELASTIC, TYPE=LAMINA")
        L.append("%.6e, %.6e, %.4g, %.6e, %.6e, %.6e"
                 % (md["E"][0], md["E"][1], md["nu"][0],
                    md["G"][0], md["G"][1], md["G"][2]))
        L.append("*DENSITY")
        L.append("%g," % md["rho"])
    L.append("*SHELL SECTION, ELSET=EALL, COMPOSITE")
    for t, m, ang in zip(thick, mats, angles):
        L.append("%.8g, 3, %s, %g" % (t, m.upper(), ang))
    L.append("*BOUNDARY")
    L.append("NROOT, 1, 6")
    L.append("NALL, 6, 6")
    L.append("*STEP, NAME=MODES")
    L.append("*FREQUENCY")
    L.append("8,")
    L.append("*END STEP")
    path = os.path.join(HERE, "ex4_%s_fsdtfreq.inp" % slug)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s" % os.path.basename(path))


if __name__ == "__main__":
    for name, (angles, mats, thick) in LAYUPS.items():
        write_fsdt_freq(name, angles, mats, thick)
