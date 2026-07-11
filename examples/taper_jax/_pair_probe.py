"""_pair_probe.py -- is web 1's top/bottom band-column PAIRING reversed at 0.2h?
For each web: the across-band direction at the top attachment vs at the bottom
(as currently paired).  dot<0 => the column lines are crossed (twisted plate)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "taper"))
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))

from taper_common import WINDIO
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import section_skeleton, build_section_mesh

sys.path.insert(0, HERE)
ThinBlade = __import__("6_thin_02h_study").ThinBlade

for label, fac in (("FULL", 1.0), ("0.2h", 0.2)):
    blade = ThinBlade(load_blade(WINDIO), fac)
    cs = build_cross_section(blade, 0.2, mesh_size=0.02)
    skel = section_skeleton([cs, cs], mesh_size=0.02, nw=3)
    sec = build_section_mesh([cs, cs], skel, nr=4)
    P = sec["stations"][0][:, :2]
    print("\n===== %s =====" % label)
    for wi, (top, bot) in enumerate(sec["wpair"]):
        def sid(i, l=4, nr=4):
            return i * (nr + 1) + l
        tdir = P[sid(top[-1])] - P[sid(top[0])]              # across-band at top (paired order)
        bdir = P[sid(bot[-1])] - P[sid(bot[0])]              # across-band at bottom (paired order)
        d = float(np.dot(tdir, bdir) / (np.linalg.norm(tdir) * np.linalg.norm(bdir) + 1e-30))
        # do the outer column lines cross?  segments top0->bot0 and topN->botN
        def cross2(o, d2):
            return d2[0] * o[1] - d2[1] * o[0]
        a0, b0 = P[sid(top[0])], P[sid(bot[0])]
        aN, bN = P[sid(top[-1])], P[sid(bot[-1])]
        # segment intersection test
        r = b0 - a0; s = bN - aN
        den = r[0] * s[1] - r[1] * s[0]
        crossed = False
        if abs(den) > 1e-30:
            t = ((aN - a0)[0] * s[1] - (aN - a0)[1] * s[0]) / den
            u = ((aN - a0)[0] * r[1] - (aN - a0)[1] * r[0]) / den
            crossed = 0.0 < t < 1.0 and 0.0 < u < 1.0
        print("web %d : across-dir dot(top,bot) = %+.4f   outer column lines cross mid-web: %s"
              % (wi, d, crossed))
