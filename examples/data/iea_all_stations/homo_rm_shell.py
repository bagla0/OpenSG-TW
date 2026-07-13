'''
homo_rm_shell.py  --  RM 6-DOF SHELL homogenization of the 1-D shell YAMLs -> Timoshenko 6x6
================================================================================================
Runs the paper's Reissner-Mindlin cross-section model (6-DOF independent-omega3 ring element with an
element-wise drilling Lagrange multiplier, gamma_23 tied by MITC) on every 1-D shell SG YAML in
<dir>, and writes the Timoshenko 6x6 stiffness (VABS order: EA, GA2, GA3, GJ, EI2, EI3) per station.

    C6 = ring_6dof(load_ring(<name>_shell.yaml))          # xsec_5v6_master (opensg_2_0 env)

The RM ring assembly is JAX under the hood (the element stiffness is vmapped over the ring elements).
vmap CANNOT batch across stations -- each station's ring has a different number of nodes/elements
(ragged) -- so stations run in a loop; use --jobs to run several as separate processes.

    ~/miniconda3/envs/opensg_2_0/bin/python homo_rm_shell.py                 # all shells in 1d_yaml/
    ...                                    homo_rm_shell.py --r 0.247        # one station
    ...                                    homo_rm_shell.py --jobs 6         # 6 stations in parallel
Output: <out>/C6_rm_<name>.txt (the 6x6) + printed diagonal per station.
================================================================================================
'''
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def find_xsec(d):
    cands = [os.path.expanduser("~/OpenSG-TW-claude/examples/TW-paper/xsec_paper")]
    x = d
    for _ in range(12):
        cands.append(os.path.join(x, "examples", "TW-paper", "xsec_paper"))
        nx = os.path.dirname(x)
        if nx == x:
            break
        x = nx
    for c in cands:
        if os.path.isfile(os.path.join(c, "xsec_5v6_master.py")):
            return c
    return None


XS = find_xsec(HERE)
if XS is None:
    sys.exit("could not find xsec_paper/xsec_5v6_master.py")
REPO = os.path.abspath(os.path.join(XS, "..", "..", ".."))
for q in (XS, REPO, os.path.join(REPO, "mitc_rm_segment")):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from xsec_5v6_master import load_ring, ring_6dof, LBL          # demo now guarded -> fast import


def process(f, out):
    nm = os.path.basename(f).replace("_shell.yaml", "").replace(".yaml", "")
    t0 = time.time()
    try:
        C6 = np.asarray(ring_6dof(load_ring(f)))
        np.savetxt(os.path.join(out, "C6_rm_%s.txt" % nm), C6)
        d = "  ".join("%s=%.4g" % (LBL[i], C6[i, i]) for i in range(6))
        print("[%-10s] %s  [%.1fs]" % (nm, d, time.time() - t0), flush=True)
    except Exception as e:
        print("[%-10s] FAIL %s" % (nm, repr(e)[:170]), flush=True)


def main():
    ap = argparse.ArgumentParser(description="RM 6-DOF shell Timoshenko 6x6 from 1-D shell YAMLs")
    ap.add_argument("--dir", default=os.path.join(HERE, "1d_yaml"))
    ap.add_argument("--r", type=float, default=None, help="single station r (else all in --dir)")
    ap.add_argument("--out", default=os.path.join(HERE, "homo_rm"))
    ap.add_argument("--glob", default="*_shell.yaml")
    ap.add_argument("--jobs", type=int, default=1, help="run this many stations in parallel processes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.dir, a.glob)))
    if a.r is not None:
        tg = "r%04d" % round(a.r * 1000)
        files = [f for f in files if tg in os.path.basename(f)]
    print("RM Shell based Timoshenko 6x6 ; %d" % len(files),
          flush=True)
    if a.jobs > 1 and len(files) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.jobs, len(files))) as pool:
            pool.starmap(process, [(f, a.out) for f in files])
    else:
        for f in files:
            process(f, a.out)
    print("\nwrote OpenSG_RM_Shell_*.txt ->", a.out)


if __name__ == "__main__":
    main()
