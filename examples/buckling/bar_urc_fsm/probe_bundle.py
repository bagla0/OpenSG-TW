"""probe_bundle.py -- what does the RM bundle already carry, so we do not recompute ABD by hand?

blade_fsm.py gets per-element ABD from a precomputed iea_sNN_abd.yaml.  No such file exists for BAR-URC, so
either (a) the bundle from solve_tw_from_yaml already holds the per-element ABD used in homogenization -- in
which case N and ABD stay consistent by construction, which is what we want -- or (b) we must build it from
the yaml layup with an OML reference.  Find out which, and confirm the reference convention.
"""
import os, sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "buckling"))
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml

SHELL15 = os.path.join(ROOT, "tests", "data", "1Dshell_15.yaml")
LOCAL15 = os.path.join(HERE, "Shell_1DSG", "1Dshell_15.yaml")

# Are the repo copy and the freshly downloaded Shell_1DSG copy the same file? The e3-sign audit was done on
# the download; the validated 0.35% dehom was run on the repo copy. If they differ, that matters.
import hashlib
for tag, p in (("repo tests/data", SHELL15), ("downloaded Shell_1DSG", LOCAL15)):
    if os.path.exists(p):
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        print("%-24s %s  md5=%s  bytes=%d" % (tag, os.path.basename(p), h, os.path.getsize(p)))
    else:
        print("%-24s MISSING %s" % (tag, p))

B = solve_tw_from_yaml(SHELL15, frac=0.0)
print("\nbundle keys and shapes (frac=0.0 -> OML reference):")
for k in sorted(B.keys()):
    v = B[k]
    try:
        a = np.asarray(v)
        print("   %-14s shape=%-18s dtype=%s" % (k, str(a.shape), a.dtype))
    except Exception:
        print("   %-14s type=%s" % (k, type(v).__name__))

for key in ("ABD", "abd", "ABD_e", "Q", "plate", "strip"):
    if key in B:
        a = np.asarray(B[key])
        print("\n%s[0] =\n%s" % (key, np.array2string(a[0] if a.ndim >= 3 else a, precision=4)))

print("\nTimo 6x6 diagonal:", np.diag(np.asarray(B["Timo"])))
print("frac recorded in bundle:", B.get("frac", "(absent)"))
