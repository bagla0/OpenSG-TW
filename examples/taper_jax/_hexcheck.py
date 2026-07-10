"""Pure-HEX IEA r=0.2->0.3 through the JAX solver: boundary must match the FEniCS
boundary numbers from the earlier deliverable (all-quad boundary, same mesh)."""
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
from opensg_jax.fe_jax.solid_taper import compute_timo_taper_solid_seg, _PERM3, _PERMF

blade = load_blade(WINDIO)
cs1 = build_cross_section(blade, 0.2, mesh_size=0.02)
cs2 = build_cross_section(blade, 0.3, mesh_size=0.02)
z1, z2 = blade_span_z(blade, 0.2), blade_span_z(blade, 0.3)
res = hex_between_sections(cs1, cs2, z1, z2, nr=4, nsp=12, nw=3, mesh_size=0.02)
oris, hmats = solid_yaml_payload(res, cs1, cs2)
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
DL, DR, DS, info = compute_timo_taper_solid_seg(seg, verbose=False)
FEN_L = [2.7005e10, 7.2199e8, 3.1419e8, 3.8167e9, 3.4176e10, 2.3867e11]
FEN_R = [2.5768e10, 6.7119e8, 2.6461e8, 2.3756e9, 2.2748e10, 1.7020e11]
FEN_S = [2.6799e10, 6.9039e8, 2.9676e8, 3.0280e9, 2.8802e10, 2.0427e11]
for nm, D, ref in (("L", DL, FEN_L), ("R", DR, FEN_R), ("SEG", DS, FEN_S)):
    d = [100.0 * (D[i, i] - ref[i]) / ref[i] for i in range(6)]
    print("%s  JAX-hex: %s" % (nm, "  ".join("%.4e" % D[i, i] for i in range(6))))
    print("%s  vs FEniCS: %s" % (nm, "  ".join("%+.2f%%" % v for v in d)))
