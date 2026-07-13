'''
homo_jax_solid.py  --  JAX-FEM 2-D SOLID homogenization of the 2d_yaml meshes -> Timoshenko 6x6
================================================================================================
Runs the JAX finite-element MSG solid cross-sectional solver on every 2-D solid SG YAML in <dir> and
writes the Timoshenko 6x6 stiffness (VABS order: EA, GA2, GA3, GJ, EI2, EI3) per station.

    C6 = compute_timo_from_yaml(<name>_solid.yaml)    # opensg_jax.fe_jax.solid_timo (opensg_2_0 env)

vmap: the per-section element assembly IS vmapped inside compute_timo_from_yaml (that's the "use jax
/ vmap" win -- the whole element loop is one vectorized batched call).  vmap CANNOT batch ACROSS
stations because each station's mesh has a different number of nodes/elements (ragged), so stations
run in a loop; the JAX runtime is warmed once and reused, so only the first station pays compile.

    ~/miniconda3/envs/opensg_2_0/bin/python homo_jax_solid.py                # all solids in 2d_yaml/
    ...                                     homo_jax_solid.py --r 0.247      # one station
Output: <out>/C6_jax_<name>.txt (the 6x6) + printed diagonal per station.
================================================================================================
'''
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def find_repo(d):
    cands = [os.path.expanduser("~/OpenSG-TW-claude")]
    x = d
    for _ in range(12):
        cands.append(x)
        nx = os.path.dirname(x)
        if nx == x:
            break
        x = nx
    for c in cands:
        if os.path.isdir(os.path.join(c, "opensg_jax")):
            return c
    return None


REPO = find_repo(HERE)
if REPO is None:
    sys.exit("could not find the OpenSG-TW repo (opensg_jax/)")
for q in (REPO, os.path.join(REPO, "opensg_jax")):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def main():
    ap = argparse.ArgumentParser(description="JAX 2-D solid Timoshenko 6x6 from 2-D solid YAMLs")
    ap.add_argument("--dir", default=os.path.join(HERE, "2d_yaml"))
    ap.add_argument("--r", type=float, default=None, help="single station r (else all in --dir)")
    ap.add_argument("--out", default=os.path.join(HERE, "homo_jax"))
    ap.add_argument("--glob", default="*_solid.yaml")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.dir, a.glob)))
    files = [f for f in files if "t1only" not in os.path.basename(f)]     # skip the t1-only variant
    if a.r is not None:
        tg = "r%04d" % round(a.r * 1000)
        files = [f for f in files if tg in os.path.basename(f)]
    print("OpenSG-JAX solid -> Timoshenko 6x6 ; %d"
          % len(files), flush=True)
    for f in files:
        nm = os.path.basename(f).replace("_solid.yaml", "").replace(".yaml", "")
        t0 = time.time()
        try:
            C6 = np.asarray(compute_timo_from_yaml(f, verbose=False))
            np.savetxt(os.path.join(a.out, "C6_jax_%s.txt" % nm), C6)
            d = "  ".join("%s=%.4g" % (LBL[i], C6[i, i]) for i in range(6))
            print("[%-10s] %s  [%.1fs]" % (nm, d, time.time() - t0), flush=True)
        except Exception as e:
            print("[%-10s] FAIL %s" % (nm, repr(e)[:170]), flush=True)
    print("\nwrote OpenSG__JAX_*.txt ->", a.out)


if __name__ == "__main__":
    main()
