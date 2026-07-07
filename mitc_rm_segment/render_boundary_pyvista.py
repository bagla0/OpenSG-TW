"""render_boundary_pyvista.py -- boundary (end cross-section) of the EXACT webbed
ellipse meshes, rendered from the real cells with PyVista: the RM shell ring (slice
of the shell surface) and the 3-D solid end face (slice of the hex mesh).  Proves
the solid is watertight at the web/skin junctions (single connected region).

    python render_boundary_pyvista.py <solid_e3w.yaml> <shell_e3w.yaml> [out_dir]
"""
import os, sys
import numpy as np
import yaml
import pyvista as pv

pv.OFF_SCREEN = True
CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
SOL, SHL = sys.argv[1], sys.argv[2]
OUTD = sys.argv[3] if len(sys.argv) > 3 else "."

# --- solid: hex UnstructuredGrid ---
d = yaml.load(open(SOL), Loader=CL)
qn = np.array([[float(v) for v in (r[0] if isinstance(r, list) else r).split()] for r in d["nodes"]])
hexes = np.array([[int(v) for v in (r[0] if isinstance(r, list) else r).split()] for r in d["elements"]]) - 1
cells = np.hstack([np.full((len(hexes), 1), 8, int), hexes]).ravel()
grid = pv.UnstructuredGrid(cells, np.full(len(hexes), pv.CellType.HEXAHEDRON, np.uint8), qn)
nreg = int(grid.connectivity().get_array("RegionId").max()) + 1
print("solid %d nodes %d hexes | connected regions = %d (1 = watertight)" % (len(qn), len(hexes), nreg))
zmin, zmax = qn[:, 2].min(), qn[:, 2].max()
sol_x = grid.slice(normal="z", origin=(0, 0, zmin + 1e-4 * (zmax - zmin)))

# --- shell: quad PolyData ---
ds = yaml.load(open(SHL), Loader=CL)
sn = np.array(ds["nodes"], float)
quads = np.array(ds["elements"], int) - 1
faces = np.hstack([np.full((len(quads), 1), 4, int), quads]).ravel()
shell = pv.PolyData(sn, faces)
shl_x = shell.slice(normal="z", origin=(0, 0, sn[:, 2].min() + 1e-4))

p = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1180, 560), border=False)
p.subplot(0, 0)
p.add_text("RM shell ring", position="upper_edge", font_size=11, color="black")
p.add_mesh(shl_x, color="#26456e", line_width=3)
p.camera_position = [(0, 0, -5), (0, 0, 0), (0, 1, 0)]; p.enable_parallel_projection()
p.camera.parallel_scale = 0.72
p.subplot(0, 1)
p.add_text("3-D solid end face", position="upper_edge", font_size=11, color="black")
p.add_mesh(sol_x, color="#e9b98a", show_edges=True, edge_color="#7a4a1e", line_width=1.0)
p.camera_position = [(0, 0, zmin - 5), (0, 0, zmin), (0, 1, 0)]; p.enable_parallel_projection()
p.camera.parallel_scale = 0.72
fn = os.path.join(OUTD, "ell_boundary_mesh.png")
p.screenshot(fn, transparent_background=False); p.close(); print("wrote", fn)
