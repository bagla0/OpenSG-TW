"""Can nw / hoop-mesh settings untangle the 0.2h web-junction section cells?"""
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

sys.path.insert(0, HERE)
ThinBlade = __import__("6_thin_02h_study").ThinBlade
blade = ThinBlade(load_blade(WINDIO), 0.2)

for nw, ms in [(3, 0.02), (2, 0.02), (1, 0.02), (3, 0.01), (2, 0.01), (1, 0.01), (1, 0.04), (2, 0.04)]:
    try:
        cs1 = build_cross_section(blade, 0.2, mesh_size=ms)
        cs2 = build_cross_section(blade, 0.3, mesh_size=ms)
        res = hex_between_sections(cs1, cs2, blade_span_z(blade, 0.2), blade_span_z(blade, 0.3),
                                   nr=4, nsp=12, nw=nw, mesh_size=ms)
        sj = _hex_min_sj(res["nodes"], np.asarray(res["hexes"]))
        ninv = int((sj <= 0).sum())
        print("nw=%d mesh=%.2f : inverted=%-4d minSJ=%+.3f %s"
              % (nw, ms, ninv, sj.min(), "CLEAN" if ninv == 0 else ""), flush=True)
    except Exception as e:
        print("nw=%d mesh=%.2f : ERROR %s" % (nw, ms, str(e)[:60]), flush=True)
