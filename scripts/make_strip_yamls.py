"""
Write the two cross-section input files for the isotropic plate-strip example
(into this Claude_code directory):

  strip_iso_1D.yaml     -- 1D shell structure-genome (straight line of nodes along
                           the width, with the wall laminate and the local frame).
                           Read by strip_RM.py / strip_Kirchhoff.py.
  strip_iso_solid.yaml  -- 2D solid cross-section (rectangle of quad elements).
                           Read by the FEniCS 2D-solid analysis strip_solid.py.

Strip: width W=1 m, thickness h = HW*W, isotropic E=70 GPa, nu=0.3.
Change HW below (0.01 thin, 0.20 thick) and re-run to regenerate both files.
"""
import os
import numpy as np
import yaml as _yaml

HERE = os.path.dirname(os.path.abspath(__file__))
W = 1.0
HW = 0.01                      # h/W (thin); set 0.20 for thick
H = HW*W
ISO = {"E": [70e9, 70e9, 70e9], "G": [26.923e9, 26.923e9, 26.923e9], "nu": [0.3, 0.3, 0.3]}
N1D = 81                       # nodes along the width for the 1D genome
NC, NR = 120, 16               # width x thickness quads for the 2D solid


def write_1d(path):
    """1D shell genome: nodes along y2 in [-W/2, W/2] at the top OML (y3=H/2);
    one chain of 2-node line elements; single isotropic ply of thickness h;
    local frame e1=beam(0,0,1), e2=width(1,0,0), e3=inward(0,-1,0)."""
    y2 = np.linspace(-W/2, W/2, N1D)
    data = {
        "nodes": [[float(y), float(H/2), 0.0] for y in y2],
        "elements": [[k+1, k+2] for k in range(N1D-1)],
        "sets": {"element": [{"name": "strip", "labels": list(range(1, N1D))}]},
        "sections": [{"elementSet": "strip", "layup": [["mat", float(H), 0.0]]}],
        "materials": [{"name": "mat", "density": 1800.0, "elastic": ISO}],
        "elementOrientations": [[0., 0., 1., 1., 0., 0., 0., -1., 0.] for _ in range(N1D-1)],
    }
    with open(path, "w") as f:
        f.write(f"# Isotropic plate-strip 1D shell genome  (W={W} m, h/W={HW}, h={H} m)\n")
        f.write("# frame per element [e1(beam) e2(width) e3(inward)]; one isotropic ply.\n")
        _yaml.safe_dump(data, f, sort_keys=False)


def write_solid(path):
    """2D solid rectangle centred at the origin: y2 in [-W/2,W/2],
    y3 in [-H/2,H/2], NC x NR quads; isotropic; e1=beam, e3=+y3."""
    y2 = np.linspace(-W/2, W/2, NC+1)
    y3 = np.linspace(-H/2, H/2, NR+1)
    nid = lambda i, k: i*(NC+1) + k
    nodes = [(y2[k], y3[i]) for i in range(NR+1) for k in range(NC+1)]
    elems, oris = [], []
    for i in range(NR):
        for k in range(NC):
            a, b = nid(i, k), nid(i, k+1)
            c, d = nid(i+1, k+1), nid(i+1, k)
            elems.append((a+1, b+1, c+1, d+1))
            oris.append([0., 0., 1., 1., 0., 0., 0., 1., 0.])
    with open(path, "w") as f:
        f.write(f"# Isotropic plate-strip 2D SOLID cross-section (W={W} m, h/W={HW}, h={H} m)\n")
        f.write("nodes:\n")
        for y2v, y3v in nodes:
            f.write(f" - [{y2v:.10f} {y3v:.10f} 0.0]\n")
        f.write("elements:\n")
        for e in elems:
            f.write(f" - [{e[0]} {e[1]} {e[2]} {e[3]}]\n")
        f.write("sets:\n  element:\n  - name: strip\n    labels: ["
                + ", ".join(str(j+1) for j in range(len(elems))) + "]\n")
        f.write("materials:\n - name: strip\n")
        f.write(f"   E: {ISO['E']}\n   G: {ISO['G']}\n   nu: {ISO['nu']}\n   rho: 1800.0\n")
        f.write("elementOrientations:\n")
        for o in oris:
            f.write(" - " + str([round(float(v), 10) for v in o]) + "\n")


if __name__ == "__main__":
    write_1d(os.path.join(HERE, "strip_iso_1D.yaml"))
    write_solid(os.path.join(HERE, "strip_iso_solid.yaml"))
    print(f"wrote strip_iso_1D.yaml ({N1D} nodes, {N1D-1} line elements) and "
          f"strip_iso_solid.yaml ({(NC+1)*(NR+1)} nodes, {NC*NR} quads) for h/W={HW}")
