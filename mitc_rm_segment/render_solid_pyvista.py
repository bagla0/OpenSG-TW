"""render_solid_pyvista.py -- render the EXACT solid mesh fed to FEniCS with PyVista
(the real hexahedra, not a sketch), to show the webbed-ellipse solid is watertight at
the web/skin junctions.  Produces an isometric 3-D view and an axial cross-section
slice.  Also prints PyVista's connected-region count.

    python render_solid_pyvista.py <solid_e3w.yaml> [out_dir]
"""
import os, sys
import numpy as np
import yaml
import pyvista as pv

pv.OFF_SCREEN = True
CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
SOL = sys.argv[1]
OUTD = sys.argv[2] if len(sys.argv) > 2 else "."

d = yaml.load(open(SOL), Loader=CL)
nodes = np.array([[float(v) for v in (r[0] if isinstance(r, list) else r).split()]
                  for r in d["nodes"]])
hexes = np.array([[int(v) for v in (r[0] if isinstance(r, list) else r).split()]
                  for r in d["elements"]]) - 1
n = len(hexes)
cells = np.hstack([np.full((n, 1), 8, int), hexes]).ravel()
ctypes = np.full(n, pv.CellType.HEXAHEDRON, np.uint8)
grid = pv.UnstructuredGrid(cells, ctypes, nodes)
print("solid mesh: %d nodes, %d hexes" % (len(nodes), n))
print("PyVista connected regions:", grid.connectivity().get_array("RegionId").max() + 1,
      "(1 = single watertight body)")

zmin, zmax = nodes[:, 2].min(), nodes[:, 2].max()

# --- isometric 3-D, beam axis (z) horizontal ---
p = pv.Plotter(off_screen=True, window_size=(1000, 620))
p.add_mesh(grid, color="#e07b39", show_edges=True, edge_color="#3a3a3a", line_width=0.4)
dvec = np.array([0.40, -0.89, 0.22]); dvec /= np.linalg.norm(dvec)
up = np.cross([0, 0, 1.0], dvec); up /= np.linalg.norm(up)
foc = np.array([0, 0, 0.5 * (zmin + zmax)])
p.enable_parallel_projection()
p.camera_position = [tuple(foc - 8 * dvec), tuple(foc), tuple(up)]
p.camera.parallel_scale = 1.15
p.add_axes(line_width=3)
f1 = os.path.join(OUTD, "solid_pv_iso.png")
p.screenshot(f1, transparent_background=False); p.close(); print("wrote", f1)

# --- axial cross-section: slice at the L end, viewed down the beam axis ---
sl = grid.slice(normal="z", origin=(0, 0, zmin + 1e-4 * (zmax - zmin)))
p = pv.Plotter(off_screen=True, window_size=(760, 560))
p.add_mesh(sl, color="#e9b98a", show_edges=True, edge_color="#7a4a1e", line_width=1.0)
p.enable_parallel_projection()
p.camera_position = [(0, 0, zmin - 5), (0, 0, zmin), (0, 1, 0)]
p.camera.parallel_scale = 0.75
f2 = os.path.join(OUTD, "solid_pv_xsec.png")
p.screenshot(f2, transparent_background=False); p.close(); print("wrote", f2)
