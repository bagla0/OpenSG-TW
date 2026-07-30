"""make_1dyaml.py -- write garg_plates.yaml: the Garg et al. (2023) benchmark laminates
as an OpenSG 1-D shell SG yaml (the systematic input route: layups + materials travel in
the yaml; the runner reads it with msg_mesh.load_yaml and calls the core msg_rm_plate).

Sections (Garg sec. 3 configurations, unit total thickness each):
  garg_A_090   [0/90/0]    Pagano material, equal thirds        (Garg figs 3-4)
  garg_A_909   [90/0/90]   Pagano material, equal thirds
  garg_B_0990  [0/90/90/0] AS4-type, equal quarters             (Garg fig 6)
  garg_B_0909  [0/90/0/90] AS4-type, equal quarters
  garg_C_sand  [0/core/0]  faces 0.1h, core 0.8h                (Garg fig 7)

Geometry is a placeholder (five disjoint unit segments): the plate homogenization and
recovery use only materials/sections; nodes exist so load_yaml parses the file as any
other 1-D SG.
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "TW-paper", "rm_thickness"))
from materials import MATDB                     # Pagano / AS4 / sandwich sets (Garg sec. 3)

CASES = [
    ("garg_A_090", [1 / 3, 1 / 3, 1 / 3], [0.0, 90.0, 0.0], ["pagano"] * 3),
    ("garg_A_909", [1 / 3, 1 / 3, 1 / 3], [90.0, 0.0, 90.0], ["pagano"] * 3),
    ("garg_B_0990", [0.25] * 4, [0.0, 90.0, 90.0, 0.0], ["as4"] * 4),
    ("garg_B_0909", [0.25] * 4, [0.0, 90.0, 0.0, 90.0], ["as4"] * 4),
    ("garg_C_sand", [0.1, 0.8, 0.1], [0.0, 0.0, 0.0], ["face", "core", "face"]),
]

nodes, elements, esets, sections = [], [], [], []
for k, (name, thk, ang, mats) in enumerate(CASES):
    nodes.append([2.0 * k, 0.0, 0.0])
    nodes.append([2.0 * k + 1.0, 0.0, 0.0])
    elements.append([2 * k + 1, 2 * k + 2])              # 1-based
    esets.append({"name": name, "labels": [k + 1]})
    sections.append({"elementSet": name,
                     "layup": [[m, float(t), float(a)] for m, t, a in zip(mats, thk, ang)]})

materials = [{"name": nm,
              "density": float(MATDB[nm].get("rho", 1.0)),
              "elastic": {"E": [float(v) for v in MATDB[nm]["E"]],
                          "G": [float(v) for v in MATDB[nm]["G"]],
                          "nu": [float(v) for v in MATDB[nm]["nu"]]}}
             for nm in ("pagano", "as4", "face", "core")]

doc = {"nodes": nodes, "elements": elements, "materials": materials,
       "sets": {"element": esets}, "sections": sections}
out = os.path.join(HERE, "garg_plates.yaml")
with open(out, "w") as f:
    yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
print("wrote", out)
