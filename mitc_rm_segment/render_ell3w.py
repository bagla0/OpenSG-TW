"""render_ell3w.py -- webbed-ellipse mesh figures (beam axis horizontal, isometric).

Skin drawn translucent so the three tapered webs read through; same camera family
as render_paper_meshes.py.

    python render_ell3w.py [out_dir]
"""
import os, sys
import numpy as np
import pyvista as pv
import yaml

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "taper_indep_study", "figs")
os.makedirs(OUTD, exist_ok=True)


def horiz_cam(p, focal=(0.0, 0.0, 1.0), dist=8.0, tilt=0.22, azim=(0.40, -0.89), scale=1.35):
    d = np.array([azim[0], azim[1], tilt]); d /= np.linalg.norm(d)
    up = np.cross(np.array([0.0, 0.0, 1.0]), d); up /= np.linalg.norm(up)
    pos = np.array(focal) - dist * d
    p.enable_parallel_projection()
    p.camera_position = [tuple(pos), tuple(focal), tuple(up)]
    p.camera.parallel_scale = scale


def load_shell(fn):
    d = yaml.safe_load(open(fn))
    nodes = np.array(d["nodes"], float)
    quads = np.array(d["elements"], int) - 1
    return nodes, quads


def load_solid(fn):
    d = yaml.safe_load(open(fn))
    nodes = np.array([[float(v) for v in (r[0] if isinstance(r, list) else r).split()]
                      for r in d["nodes"]])
    hexes = np.array([[int(v) for v in (r[0] if isinstance(r, list) else r).split()]
                      for r in d["elements"]]) - 1
    return nodes, hexes


# ---- shell: CUTAWAY (top half of skin removed) so the three webs read clearly ----
nodes, quads = load_shell(os.path.join(HERE, "out", "ell3w", "shell_48x10", "shell_e3w.yaml"))
# skin/web split: generator emits, per axial layer, 48 skin quads then 3*6 web quads
per = 48 + 3 * 6
mask = np.array([(i % per) < 48 for i in range(len(quads))])
cent = nodes[quads].mean(axis=1)
skin_keep = mask & (cent[:, 1] < 0.02)                     # bottom-half skin only
web_of = np.where(mask, -1, ((np.arange(len(quads)) % per) - 48) // 6)
p = pv.Plotter(off_screen=True, window_size=(980, 560))
fs = np.hstack([np.full((skin_keep.sum(), 1), 4), quads[skin_keep]]).ravel()
p.add_mesh(pv.PolyData(nodes, fs), color="#4c78a8",
           show_edges=True, edge_color="#3a3a3a", line_width=0.5)
for w, col in ((0, "#c23b22"), (1, "#2e8b57"), (2, "#c9a227")):
    sel = web_of == w
    fw = np.hstack([np.full((sel.sum(), 1), 4), quads[sel]]).ravel()
    p.add_mesh(pv.PolyData(nodes, fw), color=col,
               show_edges=True, edge_color="#3a3a3a", line_width=0.6)
horiz_cam(p, tilt=0.40, azim=(0.70, -0.65), scale=1.35)
p.add_axes(line_width=3)
fn = os.path.join(OUTD, "taper_ell3w_shell_mesh.png")
p.screenshot(fn, transparent_background=False); p.close(); print("wrote", fn)

# ---- solid ----
nodes, hexes = load_solid(os.path.join(HERE, "out", "ell3w", "solid_mesh", "solid_e3w.yaml"))
cells = np.hstack([np.full((len(hexes), 1), 8), hexes]).ravel()
ct = np.full(len(hexes), pv.CellType.HEXAHEDRON, np.uint8)
grid = pv.UnstructuredGrid(cells, ct, nodes)
p = pv.Plotter(off_screen=True, window_size=(980, 560))
p.add_mesh(grid, color="#e07b39", show_edges=True, edge_color="#3a3a3a", line_width=0.4)
horiz_cam(p, scale=1.35)
p.add_axes(line_width=3)
fn = os.path.join(OUTD, "taper_ell3w_solid_mesh.png")
p.screenshot(fn, transparent_background=False); p.close(); print("wrote", fn)
