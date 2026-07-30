"""make_1dyaml.py -- write mendonca_plates.yaml: the Mendonca & Ruviaro (FEAD 260 (2026)
104609) sec.-5 laminates as an OpenSG 1-D shell SG yaml.

Sections (UNIT total thickness; the runner scales ply thicknesses by H = a/aspect):
  mr_sym      [0/90/0]   equal thirds, Pagano graphite/epoxy
  mr_asym     [0/90]     equal halves
  mr_sandwich [0/core/0] faces 0.1H, transversely isotropic Pagano-1970 core

Materials are the paper's sec.-5 values and travel IN the yaml.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

MATDB_MR = {
    "mr_lam": {"E": [172.25e9, 6.89e9, 6.89e9], "G": [3.445e9, 3.445e9, 1.387e9],
               "nu": [0.25, 0.25, 0.25]},
    "mr_face": {"E": [172.25e9, 6.89e9, 6.89e9], "G": [3.445e9, 3.445e9, 1.378e9],
                "nu": [0.25, 0.25, 0.25]},
    "mr_core": {"E": [275.6e6, 275.6e6, 3445.0e6], "G": [110.24e6, 413.4e6, 413.4e6],
                "nu": [0.25, 0.02, 0.02]},
}

CASES = [
    ("mr_sym", [1 / 3, 1 / 3, 1 / 3], [0.0, 90.0, 0.0], ["mr_lam"] * 3),
    ("mr_asym", [0.5, 0.5], [0.0, 90.0], ["mr_lam"] * 2),
    ("mr_sandwich", [0.1, 0.8, 0.1], [0.0, 0.0, 0.0], ["mr_face", "mr_core", "mr_face"]),
]

nodes, elements, esets, sections = [], [], [], []
for k, (name, thk, ang, mats) in enumerate(CASES):
    nodes.append([2.0 * k, 0.0, 0.0]); nodes.append([2.0 * k + 1.0, 0.0, 0.0])
    elements.append([2 * k + 1, 2 * k + 2])
    esets.append({"name": name, "labels": [k + 1]})
    sections.append({"elementSet": name,
                     "layup": [[m, float(t), float(a)] for m, t, a in zip(mats, thk, ang)]})

materials = [{"name": nm, "density": 1.0,
              "elastic": {"E": [float(v) for v in d["E"]],
                          "G": [float(v) for v in d["G"]],
                          "nu": [float(v) for v in d["nu"]]}}
             for nm, d in MATDB_MR.items()]

doc = {"nodes": nodes, "elements": elements, "materials": materials,
       "sets": {"element": esets}, "sections": sections}
out = os.path.join(HERE, "mendonca_plates.yaml")
with open(out, "w") as f:
    yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
print("wrote", out)
