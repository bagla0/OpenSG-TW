"""homo_st15_vabs.py -- station-15 HOMOGENIZATION check FIRST (before any dehom):
RM shell Timoshenko 6x6 (OML/centroid/IML) vs the VABS .K Timoshenko 6x6, term by term.
The VABS .K 6x6 is in the SAME section/user axes as the shell (verified == the FEniCS
solid 6x6), so this is a direct comparison.  Prints %err for every non-zero term.
"""
import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml

SHELL15 = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
KF = os.path.expanduser("~/OpenSG-TW-claude/examples/data/benchmark/st15_vabs.K")


def load_vabs_timo(path):
    lines = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(lines) if "Timoshenko Stiffness Matrix" in ln)
    rows = []
    for ln in lines[i + 1:]:
        p = ln.split()
        if len(p) == 6 and all(_isnum(x) for x in p):
            rows.append([float(x) for x in p])
        if len(rows) == 6:
            break
    return np.array(rows)


def _isnum(s):
    try:
        float(s); return True
    except ValueError:
        return False


K = load_vabs_timo(KF)
J = {r: np.asarray(solve_tw_from_yaml(SHELL15, frac=f)["Timo"])
     for r, f in (("OML", 0.0), ("centroid", 0.5), ("IML", 1.0))}

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
NZ = [(i, j) for i in range(6) for j in range(i, 6) if abs(K[i, j]) > 1e-3 * abs(K).max()]

print("=" * 86)
print("Station-15 Timoshenko homogenization: RM shell vs VABS .K   (VABS order 1-ext, "
      "2/3-shear, 4-twist, 5/6-bend)")
print("=" * 86)
hdr = "%-8s %15s %11s %11s %11s   | %6s %6s %6s" % (
    "term", "VABS .K", "RM-OML", "RM-cen", "RM-IML", "OML%", "cen%", "IML%")
print(hdr); print("-" * len(hdr))
for (i, j) in NZ:
    nm = ("%s" % LBL[i]) if i == j else ("C%d%d" % (i + 1, j + 1))
    v = K[i, j]
    row = "%-8s %15.4e %11.4e %11.4e %11.4e   " % (
        nm, v, J["OML"][i, j], J["centroid"][i, j], J["IML"][i, j])
    row += "| " + " ".join("%+6.1f" % (100 * (J[r][i, j] - v) / v)
                           for r in ("OML", "centroid", "IML"))
    print(row)
for r in ("OML", "centroid", "IML"):
    fro = np.linalg.norm(J[r] - K) / np.linalg.norm(K) * 100
    print("  ||RM(%s) - VABS|| / ||VABS||  (full 6x6 Frobenius) = %.2f%%" % (r, fro))
np.savez(os.path.join(HERE, "results", "homo_st15.npz"), K=K,
         OML=J["OML"], centroid=J["centroid"], IML=J["IML"])
print("wrote results/homo_st15.npz")
