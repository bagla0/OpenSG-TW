"""loft_render_prevabs.py -- loft the PreVABS 2-D ellipse cross-section (triangles) into
the tapered 3-D segment (wedge/prism cells) with the differential a(z),b(z) taper, and
render both the tapered segment and the cross-section (solid boundary) from the real
cells with PyVista.  Also writes the lofted wedge solid YAML for FEniCS.

    python loft_render_prevabs.py <solid_xsec.yaml> <out_dir>
"""
import os, sys
import numpy as np
import yaml
import pyvista as pv

pv.OFF_SCREEN = True
CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
XS = sys.argv[1]; OUTD = sys.argv[2] if len(sys.argv) > 2 else "."
A0, A1, B0, B1, L = 1.0, 0.65, 0.60, 0.42, 2.0
NL = 10


def row(r):
    r = r[0] if (isinstance(r, list) and len(r) == 1 and isinstance(r[0], str)) else r
    return [float(v) for v in (r.split() if isinstance(r, str) else r)]


d = yaml.load(open(XS), Loader=CL)
xy = np.array([row(r)[:2] for r in d["nodes"]])             # chordwise (x in [0,2], y centered)
tris = np.array([[int(v) - 1 for v in row(e)] for e in d["elements"]])
xc = xy[:, 0] - 1.0                                          # centre the chordwise x on the section
yc = xy[:, 1]
m = len(xy)

# loft: layer s in [0,1] -> a(s),b(s); wedge = tri(bottom) + tri(top)
zc = np.linspace(0, L, NL + 1)
def sc(z):
    s = z / L
    return (A0 + (A1 - A0) * s) / A0, (B0 + (B1 - B0) * s) / B0
nodes3 = []
for z in zc:
    fa, fb = sc(z)
    for i in range(m):
        nodes3.append([xc[i] * fa, yc[i] * fb, z])
nodes3 = np.array(nodes3)
wedges = []
for i in range(NL):
    b0 = i * m; b1 = (i + 1) * m
    for t in tris:
        wedges.append([b0 + t[0], b0 + t[1], b0 + t[2], b1 + t[0], b1 + t[1], b1 + t[2]])
wedges = np.array(wedges)
print("lofted: %d nodes, %d wedges" % (len(nodes3), len(wedges)))

# ---- render tapered segment (cutaway lower half) ----
cells = np.hstack([np.full((len(wedges), 1), 6, int), wedges]).ravel()
grid = pv.UnstructuredGrid(cells, np.full(len(wedges), pv.CellType.WEDGE, np.uint8), nodes3)
cc = nodes3[wedges].mean(1)
cut = grid.extract_cells(np.where(cc[:, 1] < 0.03)[0])
p = pv.Plotter(off_screen=True, window_size=(1000, 600), border=False)
p.add_mesh(cut, color="#e07b39", show_edges=True, edge_color="#3a3a3a", line_width=0.3)
dv = np.array([0.40, -0.89, 0.22]); dv /= np.linalg.norm(dv); up = np.cross([0, 0, 1.0], dv)
p.enable_parallel_projection()
p.camera_position = [tuple(np.array([0, 0, L / 2]) - 8 * dv), (0, 0, L / 2), tuple(up / np.linalg.norm(up))]
p.camera.parallel_scale = 1.05
f1 = os.path.join(OUTD, "ell_taper_prevabs_mesh.png")
p.screenshot(f1); p.close(); print("wrote", f1)

# ---- render solid boundary cross-section (real tris) ----
faces = np.hstack([np.hstack([[3], t]) for t in tris])
g2 = pv.PolyData(np.column_stack([xc, yc, np.zeros(m)]), faces)
p = pv.Plotter(off_screen=True, window_size=(760, 560), border=False)
p.add_mesh(g2, color="#e9b98a", show_edges=True, edge_color="#7a4a1e", line_width=0.6)
p.enable_parallel_projection(); p.camera_position = "xy"; p.reset_camera()
f2 = os.path.join(OUTD, "ell_boundary_prevabs_mesh.png")
p.screenshot(f2); p.close(); print("wrote", f2)
