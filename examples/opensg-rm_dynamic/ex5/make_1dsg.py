"""make_1dsg.py -- the through-thickness 1-D SG of the Nayak-Shenoi-Moy
transient sandwich benchmark (Example 5 of Composite Structures 64 (2004)
249-267; see README.md).

Laminate (their sec. "Example 5", digits quoted from the paper):
    (0/90/0/90/core)s -- eight graphite/epoxy plies placed symmetrically
    about a PVC foam core; a/h = 10, per-face laminate 2hf/h = 0.05 (so each
    ply is 0.0125 h), core 0.9 h.  With h = 0.1524 m (the absolute thickness
    their Example 2 sets and Example 5 inherits): ply = 1.905 mm,
    core = 137.16 mm.
    Faces (Crawley graphite/epoxy):  EL = 128 GPa, ET = 11.0 GPa,
        GLT = G13 = 4.48 GPa, G23 = 1.53 GPa, nu = 0.25, rho = 1500 kg/m^3
    Core (HEREX C70.130 PVC foam):  Ec = 103.63 MPa, Gc = 50 MPa,
        nu = 0.32, rho = 130 kg/m^3

Writes sandwich_sg.yaml (5-noded quartic elements, ONE per layer -- nine
layers including the core) + sandwich_sg.png, then prints the 8x8 plate law
and the section mass.
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
    "ge": {"E": [128.0e9, 11.0e9, 11.0e9],
           "G": [4.48e9, 4.48e9, 1.53e9],
           "nu": [0.25, 0.25, 0.25], "rho": 1500.0},
    "herex": {"E": [103.63e6] * 3, "G": [50.0e6] * 3,
              "nu": [0.32, 0.32, 0.32], "rho": 130.0},
}
H = 0.1524
TPLY = 0.0125 * H
TCORE = 0.9 * H
layup = {"mat_names": ["ge"] * 4 + ["herex"] + ["ge"] * 4,
         "thick":     [TPLY] * 4 + [TCORE] + [TPLY] * 4,
         "angles":    [0.0, 90.0, 0.0, 90.0, 0.0, 90.0, 0.0, 90.0, 0.0]}
# bottom -> top: (0/90/0/90/core/90/0/90/0) = (0/90/0/90/core)s

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
print("\nsection mass rho*h = %.4f kg/m^2  (check: 2*4*%.4g*1500 + %.4g*130"
      " = %.4f)" % (rho_h, TPLY, TCORE,
                    8 * TPLY * 1500 + TCORE * 130))
