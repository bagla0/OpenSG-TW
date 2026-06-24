import os, sys
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "windio_converter"))
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section, emit_prevabs

src = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
outdir = os.path.join(CC, "windio_converter", "out", "prevabs_r050")
blade = WindIOBlade(src)
cs = build_cross_section(blade, 0.5, mesh_size=0.01)
info = emit_prevabs(cs, outdir, name="iea22_r050", mesh_size=0.01)
print("emitted:", info)
print("files:", os.listdir(outdir))
