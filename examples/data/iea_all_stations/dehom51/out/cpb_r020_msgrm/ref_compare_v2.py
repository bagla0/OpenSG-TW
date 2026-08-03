"""ref_compare_v2.py -- CORRECT both-reference homogenization comparison via the validated
oml_ring.c6 / derr path (origin-consistent, VABS-ordered), for r=0.2 AND a 51-station
health check of the center yamls (are any broken?).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
MITC = os.path.abspath(os.path.join(ROOT, "..", "..", "..", "mitc_rm_segment"))
for q in (XSEC, MITC, os.path.abspath(os.path.join(ROOT, "..", ".."))):
    sys.path.insert(0, q)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from oml_ring import load_ring_ref, c6, derr
from xsec_5v6_master import load_solid, LBL

MIDD = os.path.join(ROOT, "shell51", "1d_yaml")
OMLD = os.path.join(ROOT, "shell51", "1d_yaml_oml51")
So20 = load_solid(os.path.join(ROOT, "..", "..", "TW-paper", "iea22_blade", "data", "C6_solid_r020.txt"))

print("=== r=0.2 diag %err vs the same 2-D solid (validated c6/derr path) ===")
for tag, d, ref in (("mid-surface", MIDD, "center"), ("OML", OMLD, "oml")):
    R = load_ring_ref(os.path.join(d, "iea_s10_shell.yaml"), ref)
    e = derr(c6(R), So20)
    print("  %-12s %s" % (tag, "  ".join("%s %+6.2f" % (LBL[i], e[i]) for i in range(6))))

print("\n=== center-yaml health check, all 51 stations (nodes, any NaN/blowup in EA/GA3) ===")
bad = []
for i in range(51):
    p = os.path.join(MIDD, "iea_s%02d_shell.yaml" % i)
    if not os.path.exists(p):
        bad.append((i, "missing")); continue
    try:
        R = load_ring_ref(p, "center")
        C = c6(R)
        d = np.diag(C)
        nn = len(R["rx"])
        ok = np.all(np.isfinite(C)) and d[0] > 0 and d[2] > 0
        if not ok:
            bad.append((i, "bad C6"))
        if i % 10 == 0 or not ok:
            print("  s%02d nodes=%-4d EA=%.3e GA3=%.3e  %s" % (i, nn, d[0], d[2], "OK" if ok else "BAD"))
    except Exception as ex:
        bad.append((i, str(ex)[:50]))
print("center yamls: %d/51 valid ; problems: %s" % (51 - len(bad), bad if bad else "none"))
