"""Compute the JAX per-element curvature k22 for the mh104 connected mesh and show the WEB k22.
The web is straight -> k22 MUST be 0. If _curvature_from_corners (list-order triples) gives a large
spurious k22 on the web, that's the GJ/GA over-prediction bug."""
import os, sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_mesh import load_yaml, read_mesh, mesh_curvature

YAML = os.path.join(os.path.dirname(__file__), "shell_ref_f020_connect.yaml")
n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
nodes, cells, lpe = read_mesh(n3d, elements, e2l)
k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))
cx = np.array([nodes[c[0]][0] for c in cells]); cy = np.array([nodes[c[0]][1] for c in cells])

print("n_elements=%d  k22: min=%.3f max=%.3f mean|k22|=%.3f" % (len(cells), k22.min(), k22.max(), np.abs(k22).mean()))
print("\nWEB elements (layup_4) -- should be k22=0 (straight web):")
for i in range(len(cells)):
    if lpe[i] == "layup_4":
        print("  elem %d  x~%.3f y~%.3f  k22=%.4f  %s" % (i, cx[i], cy[i], k22[i],
              "<-- SPURIOUS!" if abs(k22[i]) > 1.0 else ""))
print("\nlargest |k22| elements:")
for i in np.argsort(-np.abs(k22))[:6]:
    print("  elem %d  layup=%s  x~%.3f y~%.3f  k22=%.4f" % (i, lpe[i], cx[i], cy[i], k22[i]))
