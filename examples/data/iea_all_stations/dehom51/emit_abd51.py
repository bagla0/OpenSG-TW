"""emit_abd51.py -- emit the per-station laminate ABD yaml (mid-ref, per unique layup) for all 51
IEA-22 stations, via emit_abd.emit_station_abd.  These are read by the dehom and the shell-buckling
tool so the ABD is computed once and reused.  Uses the EXISTING mid-ref 1-D shell yamls."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
XSEC = os.path.join(REPO, "examples", "TW-paper", "xsec_paper")
sys.path.insert(0, REPO); sys.path.insert(0, XSEC)
from emit_abd import emit_station_abd

SHELLD = os.path.join(ROOT, "shell51", "1d_yaml")
OUT = os.path.join(HERE, "out", "abd"); os.makedirs(OUT, exist_ok=True)

for i in range(51):
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    if not os.path.exists(shell):
        print("  s%02d missing" % i); continue
    tag = "iea_s%02d" % i
    try:
        out = emit_station_abd(shell, os.path.join(OUT, tag + "_abd.yaml"), station=tag, r=i / 50.0, ref="mid")
        A11 = out["layups"][0]["ABD"][0][0]
        print("  s%02d  %d layups  (layup_0 A11 = %.3e N/m)" % (i, out["n_layups"], A11))
    except Exception as e:
        print("  s%02d FAIL: %s" % (i, str(e)[:80]))
print("done -> out/abd/iea_sNN_abd.yaml")
