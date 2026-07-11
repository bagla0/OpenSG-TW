"""_offset_probe.py -- pinpoint the 0.2h web-junction collapse: for web 1's suction
band columns print EVERY quantity in the offset chain (oml, tnode, miter normal,
fscale, ring[0], ring[nr]) at FULL vs 0.2h thickness."""
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
from opensg_io.hex_loft import section_skeleton, build_station, _band_cols
from opensg_io.section_offset import miter_normals

sys.path.insert(0, HERE)
ThinBlade = __import__("6_thin_02h_study").ThinBlade

for label, fac in (("FULL", 1.0), ("0.2h", 0.2)):
    blade = ThinBlade(load_blade(WINDIO), fac)
    cs = build_cross_section(blade, 0.2, mesh_size=0.02)
    skel = section_skeleton([cs, cs], mesh_size=0.02, nw=3)
    st = build_station(cs, skel, 0, nr=4)
    bands = _band_cols(skel, 0)
    top = bands[(1, "s")]                       # web 1 suction-side band columns
    oml = st["oml"]; rings = st["rings"]; tn = st["tnode"]; fs = st["fscale"]
    m_in = miter_normals(oml)
    if isinstance(m_in, tuple):
        m_in = m_in[0]
    m_in = np.asarray(m_in)
    print("\n===== %s : web1 suction band columns (hoop idx %s) =====" % (label, list(top)))
    print("%-5s %-22s %-8s %-8s %-18s %-22s %-22s" %
          ("i", "OML(x,y)", "tnode", "fscale", "m_in(x,y)", "ring0(x,y)", "ringNR(x,y)"))
    idx = list(range(top[0] - 1, top[-1] + 2))
    for i in idx:
        print("%-5d (%9.6f,%9.6f) %8.5f %8.4f (%8.5f,%8.5f) (%9.6f,%9.6f) (%9.6f,%9.6f)"
              % (i, oml[i][0], oml[i][1], tn[i], fs[i], m_in[i][0], m_in[i][1],
                 rings[0][i][0], rings[0][i][1], rings[4][i][0], rings[4][i][1]))
    # spacing table: OML vs inner-ring spacing between adjacent band columns
    print("adjacent-column spacing (m):")
    for a, b in zip(top[:-1], top[1:]):
        d0 = np.hypot(*(oml[b] - oml[a]))
        dN = np.hypot(*(rings[4][b] - rings[4][a]))
        print("   cols %d-%d : OML %.6f  innerNR %.9f   (offset dot: t*dm=%.6f)"
              % (a, b, d0, dN, np.hypot(*((tn[b] * m_in[b]) - (tn[a] * m_in[a])))))
