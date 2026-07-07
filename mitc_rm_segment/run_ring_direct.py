"""run_ring_direct.py <shell_contour.yaml> [solid_S6.npy] -- RM-shell boundary ring 6x6
built DIRECTLY from a PreVABS 1-D contour (webbed) yaml, bypassing the 3-D-segment
extractor.  Uses the contour's own e2/e3 orientations; webs are just extra line cells
carrying the web section id.  Prints both shear schemes next to the solid diagonal.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import yaml
from segment_element import compute_k22
from solve_segment_jax import _material_by_section
from run_ring_indep import ring_indep

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


def _norm_materials(mats):
    out = []
    for mm in mats:
        if "elastic" in mm:
            out.append(mm)
        else:
            out.append({"name": mm["name"],
                        "elastic": {"E": mm["E"], "G": mm["G"], "nu": mm["nu"]}})
    return out


d = yaml.safe_load(open(sys.argv[1]))
rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
# reference the ring to the SAME axis as the solid (pass the solid's area centroid);
# PreVABS geometry is in the LE-origin chordwise frame, so without this the ring 6x6
# carries a spurious ~EA extension-bending offset coupling and a lever-armed EI/GJ.
if len(sys.argv) > 3:
    rx[:, 0] -= float(sys.argv[3]); rx[:, 1] -= float(sys.argv[4])
cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
if cells.min() == 1:
    cells = cells - 1                                       # -> 0-indexed
ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
re2, re3 = ori[:, 3:6], ori[:, 6:9]

# subdomain id per element from the element sets, keyed by section order
sections = d["sections"]; materials = _norm_materials(d["materials"])
setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
rsub = np.zeros(len(cells), dtype=int)
for grp in d["sets"]["element"]:
    si = setname_to_sec[grp["name"]]
    for lab in grp["labels"]:
        rsub[int(lab) - 1] = si

ax = 2; cross = [0, 1]                                      # contour in x-y plane, beam axis z
D_by, G_by = _material_by_section(sections, materials, center_ref=True)
k22 = compute_k22(rx[cells].mean(1), re2, re3, cells)
print("contour: %d nodes, %d cells, sections=%d, webs(cells sec>0)=%d"
      % (len(rx), len(cells), len(sections), int((rsub > 0).sum())))

np.set_printoptions(suppress=True, linewidth=160)
for sch, nm in (("full", "full-integ"), ("mitc4_g23", "g23-tied")):
    C = ring_indep(rx, cells, rsub, re3, D_by, G_by, k22, ax, cross, shear=sch)
    C = 0.5 * (C + C.T)
    print("\n=== RM-shell ring (%s) 6x6 (x1e9) ===" % nm)
    print(np.round(C / 1e9, 4))
    print("diag:", "  ".join("%s=%.4g" % (LBL[i], C[i, i] / 1e9) for i in range(6)))
    np.save(sys.argv[1].replace(".yaml", "_ring_%s.npy" % nm.split("-")[0]), C)

if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    S = np.load(sys.argv[2])
    print("\nsolid diag:", "  ".join("%s=%.4g" % (LBL[i], S[i, i] / 1e9) for i in range(6)))
