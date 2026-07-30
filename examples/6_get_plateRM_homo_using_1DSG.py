"""Example 6 - MSG-RM plate homogenization: layup -> 1-D SG mesh YAML -> 8x8 ABDG.

The through-thickness structure gene of a plate is a 1-D mesh of 5-noded (quartic)
elements, one per ply.  This example generates that mesh file from a layup with the
``segment_plate`` helper, reads it back, and homogenizes it with the core RM code
(Yu, Hodges & Volovoi, Computers & Structures 81, 2003, Sec. 4), which returns the full
8x8 RM plate law  ABDG = [[A,B,0],[B,D,0],[0,0,G]]  directly.

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

from opensg_jax.fe_jax.segment_plate import plate_sg_yaml, read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

# ------------------------------------------------- the laminate: symmetric [45/-45]s
material_db = {"gr": {"E": [172.4e9, 6.89e9, 6.89e9],   # Pagano graphite/epoxy
                      "G": [3.45e9, 1.38e9, 3.45e9],
                      "nu": [0.25, 0.25, 0.25], "rho": 1600.0}}
layup = {"mat_names": ["gr", "gr", "gr", "gr"],
         "thick": [0.002, 0.003, 0.003, 0.002],         # meters, bottom ply first
         "angles": [45.0, -45.0, -45.0, 45.0]}          # symmetric about mid-surface

# ------------------------- 1) layup -> the through-thickness 1-D SG mesh YAML
YAML = os.path.join(CC, "examples", "data", "1d_yaml", "plate_sym45_sg.yaml")
plate_sg_yaml(YAML, layup, material_db, fraction=0.5)   # 0 = OML, 0.5 = center, 1 = IML
print("wrote 1-D SG mesh:", os.path.relpath(YAML, CC))

# ------------------------- 2) the YAML -> the RM plate law (core homogenization)
sg = read_plate_sg_yaml(YAML)
r = rm_plate_msg(sg["thick"], sg["angles"], sg["mat_names"], sg["material_db"],
                 n_per_layer=sg["n_per_layer"], elem_order=sg["elem_order"],
                 fraction=sg["fraction"])

# ------------------------- 3) the 8x8
print("\nplies: %s" % ", ".join("%s(%.1fmm/%g)" % (m, 1e3 * t, a)
                                for m, t, a in zip(sg["mat_names"], sg["thick"],
                                                   sg["angles"])))
print("RM 8x8 ABDG [[A,B,0],[B,D,0],[0,0,G]]"
      " (rows 1-6: e11,e22,g12,k11,k22,k12; rows 7-8: 2g13,2g23):")
print(r["ABDG"])
print("\nUstar_rel = %.3e  (unabsorbed second-order energy)" % r["Ustar_rel"])
B = r["ABDG"][:3, 3:6]
print("max|B| = %.3e  (symmetric laminate at mid-surface -> 0)" % np.max(np.abs(B)))
