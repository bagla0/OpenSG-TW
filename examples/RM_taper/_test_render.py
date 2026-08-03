import os, sys
CC = os.path.expanduser("~/OpenSG-TW-claude")
sys.path.insert(0, os.path.join(CC, "examples", "RM_taper"))
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import matplotlib
matplotlib.use("Agg")
import _rm_common as rm
MESH = os.path.join(CC, "examples", "data", "taper_study", "meshes")
RES = os.path.join(CC, "docs", "tutorials", "_rmout")
fig = rm.render_orientation(MESH, "thin_m45_aR070", RES, title="circle thin test")
png = os.path.join(RES, "TESTRENDER.png")
fig.savefig(png, dpi=90)
print("OK wrote", png, os.path.getsize(png), "bytes")
