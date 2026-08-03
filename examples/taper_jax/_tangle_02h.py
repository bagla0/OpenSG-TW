"""Print the geometry of the tangled 0.2h web-junction 2-D cells (direct evidence)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "taper"))
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))

from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import section_skeleton, build_section_mesh

sys.path.insert(0, HERE)
ThinBlade = __import__("6_thin_02h_study").ThinBlade
blade = ThinBlade(load_blade(WINDIO), 0.2)
cs = build_cross_section(blade, 0.2, mesh_size=0.02)
skel = section_skeleton([cs, cs], mesh_size=0.02, nw=3)
sec = build_section_mesh([cs, cs], skel, nr=4)
P = sec["stations"][0][:, :2]
faces = np.asarray(sec["faces2d"]); ftag = sec["ftag"]

area = np.zeros(len(faces))
for k in range(4):
    j = (k + 1) % 4
    area += P[faces[:, k], 0] * P[faces[:, j], 1] - P[faces[:, j], 0] * P[faces[:, k], 1]
bad = np.where(area <= 0)[0]
print("2-D cells with area<=0: %d" % len(bad), flush=True)
for f in bad[:6]:
    print("\nface %d tag %s area %.3e" % (f, ftag[f], area[f]))
    for k in range(4):
        print("   node %-6d (%.6f, %.6f)" % (faces[f][k], P[faces[f][k]][0], P[faces[f][k]][1]))
