"""render_prevabs_xsec.py -- render a PreVABS 2-D solid cross-section (OpenSG solid YAML)
with PyVista to check junction mesh quality.

    python render_prevabs_xsec.py <solid_*.yaml> [out_png]
"""
import sys
import numpy as np
import yaml
import pyvista as pv

pv.OFF_SCREEN = True
CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
d = yaml.load(open(sys.argv[1]), Loader=CL)
out = sys.argv[2] if len(sys.argv) > 2 else "prevabs_xsec.png"


def row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


nodes = np.array([row(r) for r in d["nodes"]])[:, :3]
elems = [[int(v) for v in row(e)] for e in d["elements"]]
faces = []
for e in elems:
    e = [i - 1 for i in e]                     # 1-indexed
    faces.append([len(e)] + e)
faces = np.hstack([np.array(f) for f in faces])
grid = pv.PolyData(nodes, faces)
print("%d nodes %d cells | regions=%d | cell sizes=%s"
      % (len(nodes), len(elems), int(grid.connectivity().get_array("RegionId").max()) + 1,
         sorted(set(len(e) for e in elems))))

p = pv.Plotter(off_screen=True, window_size=(900, 620))
p.add_mesh(grid, color="#e9b98a", show_edges=True, edge_color="#7a4a1e", line_width=0.8)
p.enable_parallel_projection()
p.camera_position = "xy"
p.reset_camera()
p.screenshot(out); p.close(); print("wrote", out)
