"""Corner-Jacobian (concavity) audit of the 0.2h 2-D section cells + env-flag check."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "taper"))
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))

print("OPENSG_WEB_ROWS =", os.environ.get("OPENSG_WEB_ROWS"), flush=True)
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import section_skeleton, build_section_mesh
import opensg_io.hex_loft as HL
import inspect

src = inspect.getsource(HL.build_section_mesh)
print("uniform-rows patch present in build_section_mesh:", "OPENSG_WEB_ROWS" in src, flush=True)

sys.path.insert(0, HERE)
ThinBlade = __import__("6_thin_02h_study").ThinBlade
blade = ThinBlade(load_blade(WINDIO), 0.2)
cs = build_cross_section(blade, 0.2, mesh_size=0.02)
skel = section_skeleton([cs, cs], mesh_size=0.02, nw=3)
sec = build_section_mesh([cs, cs], skel, nr=4)
P = sec["stations"][0][:, :2]
faces = np.asarray(sec["faces2d"]); ftag = sec["ftag"]

# per-corner cross products (concavity): corner k jacobian ~ cross(P[k+1]-P[k], P[k-1]-P[k])
Q = P[faces]                                              # (nf, 4, 2)
bad = []
for f in range(len(faces)):
    cj = []
    for k in range(4):
        a = Q[f, (k + 1) % 4] - Q[f, k]
        b = Q[f, (k - 1) % 4] - Q[f, k]
        cj.append(a[0] * b[1] - a[1] * b[0])
    cj = np.array(cj)
    if (cj <= 0).any() != (cj <= 0).all():                # mixed signs = CONCAVE / arrowhead
        bad.append((f, ftag[f], cj))
    elif (cj <= 0).all():
        bad.append((f, ftag[f], cj))                      # fully reversed
print("2-D cells with non-positive corner jacobian: %d" % len(bad), flush=True)
for f, t, cj in bad[:8]:
    print("  face %d tag %s  corner-J %s" % (f, t, np.array2string(cj, precision=2)))
    for k in range(4):
        print("     (%.6f, %.6f)" % (Q[f, k][0], Q[f, k][1]))
