"""Scan 0.2h hex-loft inversion counts over candidate taper pairs -> pick clean ones."""
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
from opensg_io.hex_loft import hex_between_sections
from opensg_io.conformity import min_scaled_jacobian

sys.path.insert(0, HERE)
mod = __import__("6_thin_02h_study")          # reuse ThinBlade
ThinBlade = mod.ThinBlade

blade = ThinBlade(load_blade(WINDIO), 0.2)
for (r1, r2) in [(0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40),
                 (0.20, 0.23), (0.20, 0.30)]:
    try:
        cs1 = build_cross_section(blade, r1, mesh_size=0.02)
        cs2 = build_cross_section(blade, r2, mesh_size=0.02)
        res = hex_between_sections(cs1, cs2, blade_span_z(blade, r1), blade_span_z(blade, r2),
                                   nr=4, nsp=12, nw=3, mesh_size=0.02)
        msj, ninv = min_scaled_jacobian(res["nodes"], res["hexes"])
        print("r=%.2f->%.2f : inverted=%-4d minSJ=%+.3f %s"
              % (r1, r2, ninv, msj, "CLEAN" if ninv == 0 else ""), flush=True)
    except Exception as e:
        print("r=%.2f->%.2f : ERROR %s" % (r1, r2, str(e)[:70]), flush=True)
