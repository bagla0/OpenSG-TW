"""TASK B -- IEA-22 blade, 51-station RM homogenization, genuine sequential wall time.

Input : the 51 CENTER-reference 1-D shell station yamls written by shell51/gen_shell51.py
        (REFERENCE = "center", fraction 0.5):  shell51/1d_yaml/iea_sNN_shell.yaml
Entry point identical to iea_all_stations/homo_rm_shell.py:
        C6 = ring_6dof(load_ring(yaml))        # xsec_5v6_master, opensg_2_0 env

One driver process, stations strictly sequential, per-station time.perf_counter.
The FIRST station's time includes the JAX JIT compilation (fresh process).
Writes timing_taskB.json + per-station 6x6 to out_iea51_rm/.
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XS = os.path.expanduser("~/OpenSG-TW-claude/examples/TW-paper/xsec_paper")
REPO = os.path.abspath(os.path.join(XS, "..", "..", ".."))
for q in (XS, REPO, os.path.join(REPO, "mitc_rm_segment")):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

t0 = time.perf_counter()
from xsec_5v6_master import load_ring, ring_6dof  # noqa: E402
t_import = time.perf_counter() - t0

YDIR = os.path.expanduser("~/OpenSG-TW-claude/examples/data/iea_all_stations/shell51/1d_yaml")
files = sorted(glob.glob(os.path.join(YDIR, "*_shell.yaml")))
print("stations found: %d  (%s)" % (len(files), YDIR), flush=True)

OUT = os.path.join(HERE, "out_iea51_rm")
os.makedirs(OUT, exist_ok=True)
per = []
fails = []
t_all = time.perf_counter()
for f in files:
    nm = os.path.basename(f).replace("_shell.yaml", "")
    t1 = time.perf_counter()
    try:
        C6 = np.asarray(ring_6dof(load_ring(f)))
        dt = time.perf_counter() - t1
        np.savetxt(os.path.join(OUT, "C6_rm_%s.txt" % nm), C6)
        per.append(dict(station=nm, wall_s=dt, EA=float(C6[0, 0])))
        print("[%s] %.2fs  EA=%.4e" % (nm, dt, C6[0, 0]), flush=True)
    except Exception as e:
        dt = time.perf_counter() - t1
        fails.append(dict(station=nm, wall_s=dt, error=repr(e)[:300]))
        print("[%s] FAIL %.2fs  %s" % (nm, dt, repr(e)[:160]), flush=True)
total = time.perf_counter() - t_all

out = dict(n_files=len(files), n_ok=len(per), n_fail=len(fails),
           total_s=total, avg_s=total / max(len(files), 1),
           t_import=t_import, yaml_dir=YDIR, reference="center",
           per_station=per, failures=fails,
           loadavg_end=open("/proc/loadavg").read().strip())
with open(os.path.join(HERE, "timing_taskB.json"), "w") as f:
    json.dump(out, f, indent=1)
print("TASK B done: %d/%d stations ok, total %.1fs, avg %.2fs"
      % (len(per), len(files), total, total / max(len(files), 1)), flush=True)
