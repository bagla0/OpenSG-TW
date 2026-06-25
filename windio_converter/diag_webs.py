"""Diagnose the web geometry: exact nd_arc attachment x (suction vs pressure), the implied tilt from
vertical, and the node-snapped x the shell actually uses. Tells us if webs are genuinely tilted (windIO)
or near-vertical (so the shell should be forced vertical to match the PreVABS <angle>90> solid)."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "windio_converter"))
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section

blade = WindIOBlade(os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml"))
for r in (0.2, 0.3, 0.5, 0.7, 0.9):
    cs = build_cross_section(blade, r, mesh_size=0.01)
    xy = cs["xy"]; sa = cs["s_arc"]; nodes = cs["nodes"]
    print("\n=== r=%.2f  chord=%.3f ===" % (r, cs["chord"]))
    for w in cs["webs"]:
        s, e = w["s"], w["e"]
        xs = float(np.interp(s, sa, xy[:, 0])); ys = float(np.interp(s, sa, xy[:, 1]))
        xe = float(np.interp(e, sa, xy[:, 0])); ye = float(np.interp(e, sa, xy[:, 1]))
        tilt = np.degrees(np.arctan2(xe - xs, ys - ye))                 # deg from vertical
        xsn = float(nodes[w["a"]][0]); xen = float(nodes[w["b"]][0])    # node-snapped x the shell uses
        print("  %-5s s=%.3f e=%.3f | exact x_s=%+.3f x_e=%+.3f dx=%+.3f tilt=%+5.1fdeg | node x_s=%+.3f x_e=%+.3f"
              % (w["name"], s, e, xs, xe, xe - xs, tilt, xsn, xen))
