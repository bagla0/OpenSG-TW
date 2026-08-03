"""Print web band widths + skeleton breakpoints around each web at 0.2h (direct evidence)."""
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
from opensg_io.hex_loft import section_skeleton, _lam_tuple, _thick

sys.path.insert(0, HERE)
ThinBlade = __import__("6_thin_02h_study").ThinBlade

for label, fac in (("FULL", 1.0), ("0.2h", 0.2)):
    blade = ThinBlade(load_blade(WINDIO), fac)
    cs = build_cross_section(blade, 0.2, mesh_size=0.02)
    print("\n===== %s thickness =====" % label)
    for wi, w in enumerate(cs["webs"]):
        lam = _lam_tuple(cs, w["lam"])
        print("web %d  s=%.6f e=%.6f  t_lam=%.6f m  plies=%s"
              % (wi, w["s"], w["e"], _thick(lam), [(m, round(t, 5)) for m, t, a in lam]))
    skel = section_skeleton([cs, cs], mesh_size=0.02, nw=3)
    br = np.asarray(skel["breaks"][0])
    labs = None
    # print intervals narrower than 1e-5 arc + the band intervals
    for k in range(len(br) - 1):
        w_ = br[k + 1] - br[k]
        kind = skel["kinds"][k]
        if kind[0] == "band" or w_ < 1e-5:
            print("  interval %-3d %-22s [%.8f, %.8f]  width=%.3e arc (%.2e m)"
                  % (k, str(kind), br[k], br[k + 1], w_, w_ * cs["perim"]))
