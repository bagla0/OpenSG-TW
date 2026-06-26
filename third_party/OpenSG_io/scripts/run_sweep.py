"""Convert a sweep of windIO blade stations to OpenSG 1D-shell SG YAMLs (and PreVABS XMLs)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import windIO
from opensg_io import WindIOBlade, build_cross_section, emit_opensg_yaml, emit_prevabs

src = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
out = os.path.join(os.path.dirname(__file__), "..", "out", "iea22_sweep")
os.makedirs(out, exist_ok=True)
blade = WindIOBlade(src)
print("r     chord  twist  laminates webs   -> 1D-shell YAML + PreVABS XML")
for k in range(1, 20):
    r = round(0.05 * k, 3)
    if r >= 1.0:
        break
    try:
        cs = build_cross_section(blade, r, mesh_size=0.01)
        nm = "iea22_r%03d" % round(r * 100)
        emit_opensg_yaml(cs, os.path.join(out, nm + "_shell.yaml"))
        emit_prevabs(cs, os.path.join(out, nm + "_prevabs"), name=nm, mesh_size=0.01)
        print("%.2f  %5.2f  %5.1f  %4d      %d" % (r, cs["chord"], cs["twist"], len(cs["laminates"]), len(cs["webs"])))
    except Exception as e:
        print("%.2f  FAILED: %s" % (r, repr(e)[:80]))
print("wrote ->", os.path.abspath(out))
