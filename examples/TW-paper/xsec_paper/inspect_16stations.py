import os, sys, numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "third_party/OpenSG_io"))
from opensg_io import load_blade, build_cross_section
blade = load_blade(os.path.join(R, "examples/data/windio/IEA-22-280-RWT.yaml"))
STN = [0.0, 0.02, 0.0487, 0.0665, 0.0835, 0.1022, 0.1104, 0.1364, 0.1556, 0.1967, 0.247, 0.3993, 0.5336, 0.7389, 0.98, 1.0]
bf = os.path.expanduser("~/claude_tmp/barurc_FF.npy")
if os.path.exists(bf):
    FF = np.load(bf); print("barurc_FF.npy:", FF.shape, " (30x6 spanwise BAR-URC load)")
else:
    gl = sorted(__import__("glob").glob(os.path.join(R, "examples/data/2d_yaml/bar_urc_glb/*.glb")))
    print("barurc_FF.npy MISSING; bar_urc_glb has", len(gl), "glb files -> can re-extract")
print("--- build_cross_section feasibility at the 16 windIO airfoil stations ---")
for r in STN:
    try:
        cs = build_cross_section(blade, r=r)
        print("  r=%.4f  chord=%6.2f  nodes=%4d  webs=%d" % (r, cs["chord"], len(cs["nodes"]), len(cs["webs"])))
    except Exception as e:
        print("  r=%.4f  FAILED: %s" % (r, repr(e)[:70]))
