"""render_paper_meshes.py -- paper mesh figures: ISOMETRIC view with the BEAM AXIS
HORIZONTAL (left->right) so the taper is read directly along the page.

Camera: parallel projection; view direction d has a small z-component (tilt) so the
top surface is visible; view-up chosen as cross(ez, d) so the screen-right direction
is (nearly) the beam axis z.

    python render_paper_meshes.py [out_dir]
"""
import os, sys
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import taper_study as ts

OUTD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "taper_indep_study", "figs")
os.makedirs(OUTD, exist_ok=True)


def horiz_cam(p, focal=(0.0, 0.0, 1.0), dist=8.0, tilt=0.22, azim=(0.40, -0.89),
              scale=1.25):
    """Beam axis (z) horizontal on screen: d = (a, b, c) with small c, up = ez x d."""
    d = np.array([azim[0], azim[1], tilt]); d /= np.linalg.norm(d)
    up = np.cross(np.array([0.0, 0.0, 1.0]), d); up /= np.linalg.norm(up)
    pos = np.array(focal) - dist * d
    p.enable_parallel_projection()
    p.camera_position = [tuple(pos), tuple(focal), tuple(up)]
    p.camera.parallel_scale = scale


def render(tg, kind, geom, mesh_dir):
    nodes, elems, _ = ts._load_mesh(tg, kind, mesh_dir)
    if kind == "shell":
        faces = np.hstack([np.full((len(elems), 1), 4), elems]).ravel()
        grid = pv.PolyData(nodes, faces); col = "#4c78a8"
    else:
        cells = np.hstack([np.full((len(elems), 1), 8), elems]).ravel()
        ct = np.full(len(elems), pv.CellType.HEXAHEDRON, np.uint8)
        grid = pv.UnstructuredGrid(cells, ct, nodes); col = "#e07b39"
    p = pv.Plotter(off_screen=True, window_size=(980, 560))
    p.add_mesh(grid, color=col, show_edges=True, edge_color="#3a3a3a", line_width=0.6)
    horiz_cam(p, scale=1.6 if geom == "square" else 1.3)
    p.add_axes(line_width=3)
    fn = os.path.join(OUTD, "taper_%s_%s_mesh.png" % (geom, kind))
    p.screenshot(fn, transparent_background=False)
    p.close()
    print("wrote", fn)


for geom, mdir in (("circle", os.path.join(HERE, "out", "taper_study", "meshes")),
                   ("square", os.path.join(HERE, "out", "taper_square", "meshes"))):
    for kind in ("shell", "solid"):
        render("thin_m45_aR070", kind, geom, mdir)
