"""full_taper_iso.py <solid_xsec.yaml> <out.png> -- loft the PreVABS cross-section
(triangles) into the tapered 3-D wedge segment with the differential a(z),b(z) taper and
render the FULL segment (no cutaway) in isometric view.
"""
import os, sys
import numpy as np
import yaml
import pyvista as pv

pv.OFF_SCREEN = True
CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
XS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "full_taper_iso.png"
A0, A1, B0, B1, L = 1.0, 0.65, 0.60, 0.42, 2.0
NL = 24


def row(r):
    r = r[0] if (isinstance(r, list) and len(r) == 1 and isinstance(r[0], str)) else r
    return [float(v) for v in (r.split() if isinstance(r, str) else r)]


d = yaml.load(open(XS), Loader=CL)
xy = np.array([row(r)[:2] for r in d["nodes"]])
tris = np.array([[int(v) - 1 for v in row(e)] for e in d["elements"]])
# centre the chordwise x on the section centroid (PreVABS origin at LE)
xc = xy[:, 0] - xy[:, 0].mean()
yc = xy[:, 1] - xy[:, 1].mean()
m = len(xy)

zc = np.linspace(0, L, NL + 1)
def sc(z):
    s = z / L
    return (A0 + (A1 - A0) * s) / A0, (B0 + (B1 - B0) * s) / B0
nodes3 = []
for z in zc:
    fa, fb = sc(z)
    for i in range(m):
        # beam axis along world-x (horizontal); section in the (y,z) plane
        nodes3.append([z, xc[i] * fa, yc[i] * fb])
nodes3 = np.array(nodes3)
wedges = []
for i in range(NL):
    b0 = i * m; b1 = (i + 1) * m
    for t in tris:
        wedges.append([b0 + t[0], b0 + t[1], b0 + t[2], b1 + t[0], b1 + t[1], b1 + t[2]])
wedges = np.array(wedges)
print("lofted: %d nodes, %d wedges" % (len(nodes3), len(wedges)))

cells = np.hstack([np.full((len(wedges), 1), 6, int), wedges]).ravel()
grid = pv.UnstructuredGrid(cells, np.full(len(wedges), pv.CellType.WEDGE, np.uint8), nodes3)

p = pv.Plotter(off_screen=True, window_size=(1240, 780), border=False,
               lighting="three lights")
p.add_mesh(grid, color="#f0a860", show_edges=True, edge_color="#4a4a4a",
           line_width=0.3, ambient=0.35, diffuse=0.75, specular=0.1)
p.enable_parallel_projection()
p.view_isometric()
p.camera.azimuth = -20
p.camera.elevation = 12
p.reset_camera()
p.camera.zoom(1.02)
p.screenshot(OUT); p.close(); print("wrote", OUT)
