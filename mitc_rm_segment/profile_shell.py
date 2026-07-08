"""profile_shell.py -- profile the 6-DOF RM shell pipeline (extract + rings + segment).

Case: tapered square thin iso 48x10 (the ~7 s case in the paper timing table).
"""
import cProfile, pstats, io, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)

import run_indep as ri

MDIR = os.path.join(HERE, "out", "taper_square", "meshes")
RES = os.path.join(HERE, "out", "profile_shell_res")
os.makedirs(RES, exist_ok=True)
TAG = "thin_iso_aR070"

# warm run first (JAX jit compile, ffcx-free; import cost excluded)
t0 = time.time()
S1 = ri.shell_solve_lagrange(TAG, MDIR, RES)
t1 = time.time()
print("run 1 (cold jit): %.1fs" % (t1 - t0))

pr = cProfile.Profile()
pr.enable()
S2 = ri.shell_solve_lagrange(TAG, MDIR, RES)
pr.disable()
t2 = time.time()
print("run 2 (warm):     %.1fs" % (t2 - t1))

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(40)
print(s.getvalue())

import numpy as np
print("seg diag:", np.diag(np.asarray(S2)))
print("max |run1-run2|:", np.abs(np.asarray(S1) - np.asarray(S2)).max())
