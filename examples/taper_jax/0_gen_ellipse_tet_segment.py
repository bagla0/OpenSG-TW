"""0_gen_ellipse_tet_segment.py -- small tapered hollow-ELLIPSE tube segment meshed as
TETS (gmsh OCC loft + generate(3)), exported as an OpenSG solid segment YAML.  The tet
validation case for the JAX solid taper (tet4 volume batch + tri3 boundary batch):
isotropic material (frame-invariant), so elementOrientations are identity frames.

    python 0_gen_ellipse_tet_segment.py [out.yaml]
"""
import os
import sys

import numpy as np
import gmsh
import yaml


class _Flow(list):
    pass


yaml.add_representer(_Flow, lambda d, x: d.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))

A0, B0 = 1.0, 0.6                 # outer semi-axes at z=0
S1 = 0.7                          # taper ratio at z=L
T = 0.12                          # wall thickness
L = 2.0

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "ellipse_tet_seg.yaml")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ


def ring(a, b, z):
    return occ.addCurveLoop([occ.addEllipse(0, 0, z, a, b)])


outer = occ.addThruSections([ring(A0, B0, 0.0), ring(S1 * A0, S1 * B0, L)], makeSolid=True)
inner = occ.addThruSections([ring(A0 - T, B0 - T, 0.0), ring(S1 * A0 - T, S1 * B0 - T, L)], makeSolid=True)
occ.cut(outer, inner, removeObject=True, removeTool=True)
occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.35 * T)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6 * T)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.model.mesh.generate(3)
ntag, ncoord, _ = gmsh.model.mesh.getNodes()
P = ncoord.reshape(-1, 3)
id_of = {int(t): i for i, t in enumerate(ntag)}
etypes, _etg, enodes = gmsh.model.mesh.getElements(3)
tets = None
for et, en in zip(etypes, enodes):
    if et == 4:
        tets = np.array([[id_of[int(x)] for x in row] for row in en.reshape(-1, 4)])
gmsh.finalize()
used = np.unique(tets)
P = P[used]
tets = np.searchsorted(used, tets)
print("ellipse tet segment: %d nodes / %d tets" % (len(P), len(tets)))

seg = {"nodes": [], "elements": [], "sets": {"element": []}, "elementOrientations": [], "materials": []}
for p in P:
    seg["nodes"].append(_Flow(["%.10f %.10f %.10f" % (p[0], p[1], p[2])]))
for t4 in tets:
    seg["elements"].append(_Flow([" ".join(str(n + 1) for n in t4)]))
seg["sets"]["element"].append({"name": "iso", "labels": _Flow([int(k + 1) for k in range(len(tets))])})
seg["elementOrientations"] = [_Flow([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]) for _ in range(len(tets))]
seg["materials"].append({"name": "iso",
                         "E": _Flow([7.0e10, 7.0e10, 7.0e10]),
                         "G": _Flow([2.6923e10, 2.6923e10, 2.6923e10]),
                         "nu": _Flow([0.3, 0.3, 0.3]), "rho": 1800.0})
with open(out, "w") as f:
    yaml.dump(seg, f, sort_keys=False, default_flow_style=False)
print("wrote", out)
