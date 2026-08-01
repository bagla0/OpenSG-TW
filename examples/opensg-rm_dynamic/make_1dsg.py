"""make_1dsg.py -- the through-thickness 1-D SG of the transient sandwich
benchmark (see README.md for the configuration provenance).

Writes  sandwich_sg.yaml  (5-noded quartic elements, ONE per layer -- face,
core, face -- exactly the layout the OpenSG-RM homogenization consumes) and
sandwich_sg.png (the mesh figure), then prints the 8x8 plate law and the
section mass so the deck generator's numbers can be checked by eye.

Variables
---------
MATERIAL_DB   the sandwich stiffness set (Garg et al. 2023, = the archive's
              statically validated garg_caseC data) + the dynamics densities
              (faces 1600 kg/m^3, core 100 kg/m^3 -- see README provenance)
H             total thickness 0.05 m (a/h = 10 with a = 0.5 m)
layup         [0 / core / 0], faces 0.1 H each, core 0.8 H, bottom ply first
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from opensg_jax.fe_jax.segment_plate import plate_sg_yaml, read_plate_sg_yaml, \
    plot_plate_sg
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

MATERIAL_DB = {
    "face": {"E": [131.0e9, 10.34e9, 10.34e9],
             "G": [6.205e9, 6.205e9, 3.0e9],
             "nu": [0.22, 0.22, 0.22], "rho": 1600.0},
    "core": {"E": [0.5776e9] * 3, "G": [0.1079e9] * 3,
             "nu": [0.0025] * 3, "rho": 100.0},
}
H = 0.05
layup = {"mat_names": ["face", "core", "face"],
         "thick":     [0.1 * H, 0.8 * H, 0.1 * H],
         "angles":    [0.0, 0.0, 0.0]}               # bottom ply first

yml = os.path.join(HERE, "sandwich_sg.yaml")
plate_sg_yaml(yml, layup, MATERIAL_DB, fraction=0.5)
png = plot_plate_sg(yml)
print("wrote %s + %s" % (os.path.basename(yml), os.path.basename(png)))

inp = read_plate_sg_yaml(yml)
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")
print("\nOpenSG-RM 8x8 ABDG (rows/cols: %s):" % ", ".join(ROWS))
for name, row in zip(ROWS, np.asarray(r["ABDG"])):
    print("%5s " % name + " ".join("%12.4e" % v for v in row))
rho_h = sum(inp["material_db"][m]["rho"] * t
            for m, t in zip(inp["mat_names"], inp["thick"]))
print("\nsection mass rho*h = %.4f kg/m^2  (README check: 20.0)" % rho_h)
