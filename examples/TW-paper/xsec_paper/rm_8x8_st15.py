"""rm_8x8_st15.py -- assemble and store the MSG Reissner-Mindlin 8x8 plate stiffness
[[A,B,0],[B,D,0],[0,0,G]] for every wall laminate of station 15, plus the transverse-
shear shear-flow shapes used by the dehomogenization.  Writes a human-readable .dat.
"""
import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from opensg_jax.fe_jax.msg_mesh import load_yaml
from opensg_jax.fe_jax.msg_materials import compute_ABD_matrix, plate_8x8
from opensg_jax.fe_jax.msg_transverse_shear import transverse_shear_stiffness

SHELL15 = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
DAT = os.path.join(REPO, "examples", "data", "benchmark", "st15_rm_plate_8x8.dat")

_nodes, _elem, material_db, layup_db, _e2l = load_yaml(SHELL15)


def _fmt(M):
    return "\n".join("   " + "  ".join("% .8E" % v for v in row) for row in M)


lines = ["# MSG Reissner-Mindlin plate 8x8 stiffness  [[A,B,0],[B,D,0],[0,0,G]]",
         "# Station 15 (BAR-URC) wall laminates.  Plate strain order",
         "#   [eps11, eps22, 2eps12, kappa11, kappa22, 2kappa12 | 2gamma13, 2gamma23].",
         "# A = D6[0:3,0:3] (membrane), B = D6[0:3,3:6], D = D6[3:6,3:6] (bending),",
         "# G = 2x2 transverse-shear block (MSG complementary-energy shear flow, no",
         "# shear-correction factor).  Reference surface = OML (bottom face, z=0).",
         "# Units: A,B,D SI (N/m, N, N.m); G in N/m.", ""]

for ln, info in layup_db.items():
    th, an, mn = info["thick"], info["angles"], info["mat_names"]
    D6 = np.asarray(compute_ABD_matrix(th, an, mn, material_db)[0])
    Gmat, recover, aux = transverse_shear_stiffness(th, an, mn, material_db)
    P8 = plate_8x8(D6, Gmat)
    htot = float(sum(th))
    lines += ["=" * 74,
              "LAYUP  %s   (%d plies, total thickness %.5f m)" % (ln, len(th), htot),
              "  plies (OML->IML):  " +
              "  ".join("%s@%g/%.4g" % (mn[k], an[k], th[k]) for k in range(len(th))),
              "  RM 8x8 plate stiffness:",
              _fmt(P8),
              "  transverse-shear block G (2x2, N/m):",
              _fmt(Gmat), ""]

os.makedirs(os.path.dirname(DAT), exist_ok=True)
open(DAT, "w").write("\n".join(lines))
print("wrote", DAT, "(%d laminates)" % len(layup_db))
# echo the spar-cap G as a sanity print
for ln, info in layup_db.items():
    G = transverse_shear_stiffness(info["thick"], info["angles"], info["mat_names"], material_db)[0]
    print("  %-10s  G11=%.4e  G22=%.4e  G12=%.4e" % (ln, G[0, 0], G[1, 1], G[0, 1]))
