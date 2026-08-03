"""0.2h r=0.2->0.3: ALL-HEX solid vs MIXED (hex+tet-web) solid -- is GA3 tet-degraded?"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "taper"))
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))

from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, compute_timo_taper_solid_seg,
                                           _PERM3, _PERMF)

sys.path.insert(0, HERE)
ThinBlade = __import__("6_thin_02h_study").ThinBlade
blade = ThinBlade(load_blade(WINDIO), 0.2)
cs1 = build_cross_section(blade, 0.2, mesh_size=0.02)
cs2 = build_cross_section(blade, 0.3, mesh_size=0.02)
res = hex_between_sections(cs1, cs2, blade_span_z(blade, 0.2), blade_span_z(blade, 0.3),
                           nr=4, nsp=12, nw=3, mesh_size=0.02)
oris, hmats = solid_yaml_payload(res, cs1, cs2)
web = np.array([t[0] == "web" for t in res["htag"]])
mat_names = sorted(set(hmats)); name_ix = {n: i for i, n in enumerate(mat_names)}
mp = []
for n in mat_names:
    e = _mat_block(blade, n)["elastic"]
    mp.append([e["E"][0], e["E"][1], e["E"][2], e["G"][0], e["G"][1], e["G"][2],
               e["nu"][0], e["nu"][1], e["nu"][2]])
seg = dict(nodes=np.asarray(res["nodes"])[:, _PERM3],
           batches={"hex8": (np.asarray(res["hexes"]),
                             np.array([name_ix[m] for m in hmats], int),
                             np.asarray(oris)[:, _PERMF])},
           mat_param=np.array(mp), nelem=len(res["hexes"]))
NAMES = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
for lab, sg in (("ALL-HEX", seg), ("MIXED", split_batches_to_tets(seg, mask=web))):
    DL, DR, DS, info = compute_timo_taper_solid_seg(sg, verbose=False)
    print("%s seg diag: %s" % (lab, "  ".join("%s=%.4e" % (NAMES[i], DS[i, i]) for i in range(6))),
          flush=True)
