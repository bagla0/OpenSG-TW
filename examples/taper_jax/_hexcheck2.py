"""Apples-to-apples: FEniCS compute_stiffness(Taper=True) on the SAME r=0.2->0.3 hex
segment YAML -- its Deff_l/Deff_r are computed from the segment's own end submeshes,
directly comparable to the JAX extracted boundaries."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "taper"))
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))

from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload
from opensg_io.mesh3d import export_solid_yaml
from opensg_jax.fe_jax.solid_taper import compute_timo_taper_solid_seg, _PERM3, _PERMF

OUT = os.path.join(HERE, "out_iea_taper")
os.makedirs(OUT, exist_ok=True)
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

# --- JAX on the in-memory hex segment
seg = dict(nodes=np.asarray(res["nodes"])[:, _PERM3],
           batches={"hex8": (np.asarray(res["hexes"]),
                             np.array([name_ix[m] for m in hmats], int),
                             np.asarray(oris)[:, _PERMF])},
           mat_param=np.array(mp), nelem=len(res["hexes"]))
DLj, DRj, DSj, _ = compute_timo_taper_solid_seg(seg, verbose=False)

# --- FEniCS on the SAME mesh via YAML
p = os.path.join(OUT, "hexseg_r020_030.yaml")
sets = {"element": [{"name": m, "labels": [k + 1 for k, hm in enumerate(hmats) if hm == m]}
                    for m in mat_names]}
mats = [{"name": m, **{k: _mat_block(blade, m)["elastic"][k] for k in ("E", "G", "nu")},
         "rho": _mat_block(blade, m)["density"]} for m in mat_names]
export_solid_yaml(p, res["nodes"], res["hexes"], "hex", oris, mats, sets=sets)

from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
os.chdir(OUT)
sm = SolidSegmentMesh(p)
mpz, _den = sm.material_database
t0 = time.time()
S, V0, V1s, DLf, DRf = compute_stiffness(mpz, sm.meshdata, sm.left_submesh,
                                         sm.right_submesh, Taper=True)
print("FEniCS %.1f s" % (time.time() - t0))
sym = lambda M: 0.5 * (np.asarray(M) + np.asarray(M).T)
DLf, DRf, Sf = sym(DLf), sym(DRf), sym(S)
for nm, J, F in (("L", DLj, DLf), ("R", DRj, DRf), ("SEG", DSj, Sf)):
    d = [100.0 * (J[i, i] - F[i, i]) / F[i, i] for i in range(6)]
    print("%s FEniCS : %s" % (nm, "  ".join("%.4e" % F[i, i] for i in range(6))))
    print("%s JAXvsF : %s" % (nm, "  ".join("%+.3f%%" % v for v in d)))
