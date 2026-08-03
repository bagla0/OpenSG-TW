"""1d_sg.py -- layup_db.yaml  ->  1dsg.yaml  ->  <layup_db>.out

    input   layup_db.yaml   the user's laminate: fraction, materials
                            (with density), stacking sequence, and
                            model = 0 (classical ABD) or 1 (8x8 ABDG)
    mesh    1dsg.yaml       the through-thickness 1-D SG, written by
                            segment_plate.plate_sg_yaml (+ 1dsg.png)
    output  <layup_db>.out  the OpenSG plate homogenization

Run:  python 1d_sg.py [layup_db.yaml]
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from opensg_jax.fe_jax.segment_plate import plate_sg_yaml, read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")

# ---- input -----------------------------------------------------------------
DB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "layup_db.yaml")
db = yaml.safe_load(open(DB))
model = int(db["model"])                 # 0 = ABD only, 1 = ABD + shear
fraction = float(db["fraction"])         # no code-side default: the YAML is
                                         # the ONLY place these are set
mesh = db.get("mesh") or {}
n_per_layer = int(mesh.get("n_per_layer", 1))
elem_order = int(mesh.get("elem_order", 4))

material_db = {k: {"E": [float(v) for v in m["E"]],
                   "G": [float(v) for v in m["G"]],
                   "nu": [float(v) for v in m["nu"]],
                   "rho": float(m["rho"]),          # density -> section mass
                   "full_name": m.get("full_name") or k}
               for k, m in db["materials"].items()}
layup = {"mat_names": [p["material"] for p in db["layup"]],
         "thick": [float(p["thickness"]) for p in db["layup"]],
         "angles": [float(p.get("angle", 0.0)) for p in db["layup"]]}

# ---- 1-D SG mesh, straight through segment_plate ---------------------------
yml = os.path.join(HERE, "1dsg.yaml")
plate_sg_yaml(yml, layup, material_db, n_per_layer=n_per_layer,
              elem_order=elem_order, fraction=fraction)

# ---- homogenize ------------------------------------------------------------
inp = read_plate_sg_yaml(yml)
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], n_per_layer=n_per_layer,
                 elem_order=elem_order, fraction=fraction)
n = 6 if model == 0 else 8
M = np.asarray(r["A6"] if model == 0 else r["ABDG"])
rho_h = sum(inp["material_db"][m]["rho"] * t
            for m, t in zip(inp["mat_names"], inp["thick"]))

# ---- output ----------------------------------------------------------------
out = os.path.splitext(DB)[0] + "_plate_homo.out"
with open(out, "w") as f:
    f.write("OpenSG plate homogenization of %s\n"
            % os.path.basename(yml))
    f.write("%d plies, h = %.6f m, reference fraction = %g\n"
            % (len(inp["thick"]), sum(inp["thick"]), fraction))
    f.write("model %d: %s\n\n"
            % (model, "classical 6x6 ABD" if model == 0
               else "shear-refined 8x8 ABDG"))
    f.write("rows/cols: %s\n" % ", ".join(ROWS[:n]))
    for row in M:
        f.write(" ".join("%14.6e" % v for v in row) + "\n")
    f.write("\nsection mass rho*h = %.6f kg/m^2\n" % rho_h)

print("%s + %s  ->  %s" % (os.path.basename(DB), os.path.basename(yml),
                           os.path.basename(out)))
