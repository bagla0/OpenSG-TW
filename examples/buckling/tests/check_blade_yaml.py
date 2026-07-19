"""Validate the emitted full-blade shell yaml parses and has consistent structure."""
import os, yaml
D = os.path.join(os.path.dirname(__file__), "..", "data", "iea22_blade_shell.yaml")
d = yaml.safe_load(open(D))
print("keys:", list(d.keys()))
print("nodes", len(d["nodes"]), "elems", len(d["elements"]), "ori", len(d["elementOrientations"]),
      "sections", len(d["sections"]), "materials", len(d["materials"]))
# every element in exactly one set
labels = [l for g in d["sets"]["element"] for l in g["labels"]]
print("set labels:", len(labels), "unique:", len(set(labels)), "== elems:", len(labels) == len(d["elements"]))
s = d["sections"][80]
print("sample section:", s["elementSet"], "->", s["layup"][:2])
print("materials:", [m["name"] for m in d["materials"]])
# orientation orthonormal check on element 0
import numpy as np
o = np.array(d["elementOrientations"][0]).reshape(3, 3)
print("ori[0] e1.e2=%.1e e1.e3=%.1e |e1|=%.4f" % (o[0] @ o[1], o[0] @ o[2], np.linalg.norm(o[0])))
