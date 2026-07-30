"""Example 6 - MSG-RM plate homogenization: the 8x8 ABDG from a 1-D SG mesh YAML.

Reads the through-thickness 1-D SG of a plate wall (5-noded quartic elements, one per
ply; a symmetric [45/-45]s graphite/epoxy laminate), collects the layup information from
the file, and homogenizes it with the core RM code (Yu, Hodges & Volovoi, Computers &
Structures 81, 2003, Sec. 4), which returns the full 8x8 RM plate law
ABDG = [[A,B,0],[B,D,0],[0,0,G]] directly.

The mesh YAML is generated from a LAYUP DICTIONARY by the ``segment_plate`` helper
(``plate_sg_yaml(path, layup, material_db)``, and ``plot_plate_sg`` for the companion
mesh PNG) -- already done for the shipped file.  A plate SG is always built from a layup:
it is the wall discretised through its thickness, never a cross-section contour, so no
airfoil or other geometry is involved anywhere in this example.

Run:
    python examples/6_get_plateRM_homo_using_1DSG.py
"""
import os
import sys

import numpy as np

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
np.set_printoptions(precision=4, linewidth=150)

from opensg_jax.fe_jax.segment_plate import read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

YAML = os.path.join(CC, "examples", "data", "plate_sg", "plate_sym45_sg.yaml")

# ------------------------------------------ read the 1-D SG: layup + mesh + materials
sg = read_plate_sg_yaml(YAML)
print("1-D SG : %s" % os.path.relpath(YAML, CC))
print("plies  : %s   (h = %.4f m, reference fraction = %.2f)"
      % (", ".join("%s(%.1fmm/%g)" % (m, 1e3 * t, a)
                   for m, t, a in zip(sg["mat_names"], sg["thick"], sg["angles"])),
         sum(sg["thick"]), sg["fraction"]))

# ------------------------------------------------------------- RM homogenization
r = rm_plate_msg(sg["thick"], sg["angles"], sg["mat_names"], sg["material_db"],
                 n_per_layer=sg["n_per_layer"], elem_order=sg["elem_order"],
                 fraction=sg["fraction"])

print("\nRM 8x8 ABDG [[A,B,0],[B,D,0],[0,0,G]]"
      " (rows 1-6: e11,e22,g12,k11,k22,k12; rows 7-8: 2g13,2g23):")
print(r["ABDG"])
