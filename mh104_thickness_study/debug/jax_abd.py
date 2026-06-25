"""Compute the JAX per-layup plate ABD (6x6) for the mh104 f=0.2 layups at the REAL E=10 Pa gelcoat,
to compare against FEniCS. Order [A11,A22,A66 | D11,D22,D66] on the diagonal."""
import os, sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix

YAML = os.path.join(os.path.dirname(__file__), "shell_ref_f020_connect.yaml")
n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
np.set_printoptions(precision=5, linewidth=170, suppress=False)
for ln in sorted(layup_db.keys()):
    i = layup_db[ln]
    abd = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
    np.savetxt(os.path.join(os.path.dirname(__file__), "abd_jax_%s.txt" % ln), abd)
    print("=== %s  mats=%s angles=%s ===" % (ln, i["mat_names"], i["angles"]))
    print("  A11=%.6e A22=%.6e A66=%.6e  D11=%.6e D22=%.6e D66=%.6e  A16=%.4e B11=%.4e"
          % (abd[0, 0], abd[1, 1], abd[2, 2], abd[3, 3], abd[4, 4], abd[5, 5], abd[0, 2], abd[0, 3]))
