"""make_1dsg.py -- the through-thickness 1-D SG of the Nayak-Shenoi-Moy
transient sandwich benchmark (Example 5 of Composite Structures 64 (2004)
249-267; see README.md).

Laminate (their sec. "Example 5", digits quoted from the paper):
    (0/90/0/90/core)s -- eight graphite/epoxy plies placed symmetrically
    about a PVC foam core.  The paper gives only RATIOS: h/a = 0.10 (printed
    there as "a/h is 0.10", which cannot be meant literally) and
    2 h_f / h = 0.05, where h_f is ONE face sheet -- so both faces together
    are 0.05 h, the eight plies are 0.05h/8 = 0.00625 h each, and the core
    is the remaining 0.95 h.  With h = 0.1524 m (the absolute thickness
    their Example 2 sets and Example 5 inherits): ply = 0.9525 mm,
    core = 144.78 mm, a = b = 10 h = 1.524 m.
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

from opensg_jax.fe_jax.segment_plate import plate_sg_yaml, read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

MATERIAL_DB = {
    "ge": {"E": [128.0e9, 11.0e9, 11.0e9],
           "G": [4.48e9, 4.48e9, 1.53e9],
           "nu": [0.25, 0.25, 0.25], "rho": 1500.0,
           "full_name": "graphite/epoxy face ply"},
    "herex": {"E": [103.63e6] * 3, "G": [50.0e6] * 3,
              "nu": [0.32, 0.32, 0.32], "rho": 130.0,
              "full_name": "HEREX C70.130 PVC foam core"},
}
# --- the paper's Example-5 geometry, as ALGEBRA on its two stated ratios ---
# "For this example, [h/a] is 0.10 and 2 h_f / h = 0.05 where 2 h_f is overall
#  thickness of the face plate", with "face sheets of eight plies of GE placed
#  symmetrically about a PVC foam core".
# h_f = ONE face sheet, so 2 h_f = BOTH faces together = FACE_FRAC * h, the
# eight plies split 4/4 between them, and whatever is left is core:
#     t_ply  = FACE_FRAC * h / N_PLY   = (0.05/8) h = 0.00625 h
#     t_core = (1 - FACE_FRAC) * h     = 0.95 h
# so that N_PLY * t_ply + t_core = h identically.
H = 0.1524                              # h [m]; = 6 in, set by their Example 2
FACE_FRAC = 0.05                        # 2 h_f / h, BOTH face sheets
N_PLY = 8                               # eight GE plies, four per face
TPLY = FACE_FRAC * H / N_PLY
TCORE = (1.0 - FACE_FRAC) * H
assert abs(N_PLY * TPLY + TCORE - H) < 1e-12
layup = {"mat_names": ["ge"] * 4 + ["herex"] + ["ge"] * 4,
         "thick":     [TPLY] * 4 + [TCORE] + [TPLY] * 4,
         "angles":    [0.0, 90.0, 0.0, 90.0, 0.0, 90.0, 0.0, 90.0, 0.0]}
# bottom -> top: (0/90/0/90/core/90/0/90/0) = (0/90/0/90/core)s

yml = os.path.join(HERE, "1d_sg.yaml")
plate_sg_yaml(yml, layup, MATERIAL_DB, fraction=0.5)    # writes the yaml AND the png
print("wrote %s + %s" % (os.path.basename(yml), "1d_sg.png"))

inp = read_plate_sg_yaml(yml)
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")
print("\nOpenSG-RM 8x8 ABDG (rows/cols: %s):" % ", ".join(ROWS))
for name, row in zip(ROWS, np.asarray(r["ABDG"])):
    print(" ".join("%12.4e" % v for v in row))
rho_h = sum(inp["material_db"][m]["rho"] * t
            for m, t in zip(inp["mat_names"], inp["thick"]))
rho_f, rho_c = MATERIAL_DB["ge"]["rho"], MATERIAL_DB["herex"]["rho"]
print("\nt_ply = %.4f mm, t_core = %.4f mm, sum = %.4f mm (= h)"
      % (1e3 * TPLY, 1e3 * TCORE, 1e3 * (N_PLY * TPLY + TCORE)))
print("section mass rho*h = %.4f kg/m^2  (check: %d*%.4g*%g + %.4g*%g"
      " = %.4f)" % (rho_h, N_PLY, TPLY, rho_f, TCORE, rho_c,
                    N_PLY * TPLY * rho_f + TCORE * rho_c))
