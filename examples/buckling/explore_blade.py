"""Probe the IEA-22 blade: webs per station (where they vanish), ABD yaml availability, layups per station.
Informs the conformal-mesh ABD assignment for blade_buckling.py."""
import os, sys, glob
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BUCK, "..", ".."))
IO = os.path.join(ROOT, "third_party", "OpenSG_io")
XSEC = os.path.join(ROOT, "examples", "TW-paper", "xsec_paper")
IEA = os.path.join(ROOT, "examples", "data", "iea_all_stations")
sys.path.insert(0, ROOT); sys.path.insert(0, IO); sys.path.insert(0, XSEC)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import yaml
from opensg_io import load_blade

WINDIO = os.path.join(IEA, "IEA-22-280-RWT.yaml")
SHELLD = os.path.join(IEA, "shell51", "1d_yaml")
blade = load_blade(WINDIO)

print("=== webs per station (r, #webs, [s,e] fractions) ===")
for i in range(51):
    r = i / 50.0
    try:
        w = blade.webs_at(r)
        s = ["(%.3f,%.3f)" % (x["s"], x["e"]) for x in w]
        print("  s%02d r=%.2f  nwebs=%d  %s" % (i, r, len(w), " ".join(s)))
    except Exception as e:
        print("  s%02d r=%.2f  ERR %s" % (i, r, str(e)[:50]))

print("\n=== ABD yaml availability ===")
for sub in ["abd"]:
    p = os.path.join(SHELLD, sub)
    n = len(glob.glob(os.path.join(p, "*.yaml"))) if os.path.isdir(p) else -1
    print("  %s : %d yamls" % (p, n))
p2 = os.path.join(IEA, "dehom51", "out", "abd")
n2 = len(glob.glob(os.path.join(p2, "*.yaml"))) if os.path.isdir(p2) else -1
print("  %s : %d yamls" % (p2, n2))

print("\n=== layups per station (from shell yaml sections) ===")
for i in [0, 5, 10, 25, 45, 50]:
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    if not os.path.exists(shell):
        print("  s%02d MISSING" % i); continue
    d = yaml.safe_load(open(shell))
    secs = [s["elementSet"] for s in d["sections"]]
    print("  s%02d : %d sections %s" % (i, len(secs), secs))
