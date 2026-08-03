"""Where are the 0.2h inverted hexes -- web / skin / TE?  And does PRISMATIC fold?"""
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
from opensg_io.hex_loft import hex_between_sections, _hex_min_sj
from opensg_io.conformity import min_scaled_jacobian

sys.path.insert(0, HERE)
ThinBlade = __import__("6_thin_02h_study").ThinBlade
blade = ThinBlade(load_blade(WINDIO), 0.2)

for nm, r1, r2 in (("TAPER 0.2->0.3", 0.2, 0.3), ("PRISMATIC 0.2", 0.2, None)):
    cs1 = build_cross_section(blade, r1, mesh_size=0.02)
    cs2 = cs1 if r2 is None else build_cross_section(blade, r2, mesh_size=0.02)
    z1 = blade_span_z(blade, r1)
    z2 = z1 + 2.0 if r2 is None else blade_span_z(blade, r2)
    res = hex_between_sections(cs1, cs2, z1, z2, nr=4, nsp=12, nw=3, mesh_size=0.02)
    sj = _hex_min_sj(res["nodes"], np.asarray(res["hexes"]))
    bad = np.where(sj <= 0)[0]
    tags = [res["htag"][k] for k in bad]
    kinds = {}
    for t in tags:
        key = t[0] if t[0] == "web" else ("skin(set%s,layer%s)" % (t[1], t[2]))
        kinds[key] = kinds.get(key, 0) + 1
    print("%s : inverted=%d  by kind: %s" % (nm, len(bad), kinds), flush=True)
    if len(bad):
        # hoop location: which arc positions (s) do the bad skin cells sit at?
        sec = res["sec"]
        nf = len(sec["faces2d"])
        fids = sorted(set(int(k % nf) for k in bad))
        print("   distinct 2-D cells involved: %d  (first few face ids: %s)"
              % (len(fids), fids[:12]), flush=True)
