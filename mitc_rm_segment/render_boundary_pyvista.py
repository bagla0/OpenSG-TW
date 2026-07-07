"""render_boundary_pyvista.py -- render the EXACT webbed-ellipse meshes fed to the
solvers with PyVista (real cells, no sketch): shell (cutaway so the three webs show)
and 3-D solid (isometric into the open end, webs visible joining the skin).  Prints
the solid connected-region count (1 = watertight).

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

# camera looking into the open (large) end, beam axis (z) receding
DVEC = np.array([0.40, -0.89, 0.22]); DVEC /= np.linalg.norm(DVEC)
UP = np.cross([0, 0, 1.0], DVEC); UP /= np.linalg.norm(UP)


def cam(p, focz, dist=8.0, scale=0.95):
    foc = np.array([0, 0, focz])
    p.enable_parallel_projection()
    p.camera_position = [tuple(foc - dist * DVEC), tuple(foc), tuple(UP)]
    p.camera.parallel_scale = scale


# ---- solid: real hexahedra ----
d = yaml.load(open(SOL), Loader=CL)
qn = np.array([[float(v) for v in (r[0] if isinstance(r, list) else r).split()] for r in d["nodes"]])
hexes = np.array([[int(v) for v in (r[0] if isinstance(r, list) else r).split()] for r in d["elements"]]) - 1
cells = np.hstack([np.full((len(hexes), 1), 8, int), hexes]).ravel()
grid = pv.UnstructuredGrid(cells, np.full(len(hexes), pv.CellType.HEXAHEDRON, np.uint8), qn)
print("solid %d nodes %d hexes | connected regions = %d (1 = watertight)"
      % (len(qn), len(hexes), int(grid.connectivity().get_array("RegionId").max()) + 1))
zc = 0.5 * (qn[:, 2].min() + qn[:, 2].max())

# ---- shell: real quads, cutaway (drop top-half skin so the webs show) ----
ds = yaml.load(open(SHL), Loader=CL)
sn = np.array(ds["nodes"], float)
quads = np.array(ds["elements"], int) - 1
cent = sn[quads].mean(1)
nlq = quads.shape[0]
per = int(round(nlq / (len(np.unique(qn[:, 2])) - 1)))   # quads per axial layer (approx)
# skin quads are the first 48 per axial band; identify webs by |x| spread ~0 hoop step
# robust split: a quad is "web" if all its nodes share (nearly) one x-plane
isweb = np.array([np.ptp(sn[q][:, 0]) < 1e-6 for q in quads])
keep_skin = (~isweb) & (cent[:, 1] < 0.02)
faces_keep = np.vstack([quads[keep_skin], quads[isweb]])
fk = np.hstack([np.full((len(faces_keep), 1), 4, int), faces_keep]).ravel()
shell = pv.PolyData(sn, fk)

p = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1240, 560), border=False)
p.subplot(0, 0)
p.add_text("RM shell mesh (cutaway)", position="upper_edge", font_size=11, color="black")
p.add_mesh(shell, color="#4c78a8", show_edges=True, edge_color="#26456e", line_width=0.6)
cam(p, zc)
p.subplot(0, 1)
p.add_text("3-D solid mesh (cutaway)", position="upper_edge", font_size=11, color="black")
# keep whole hexes in the lower half (centroid y < small): removes the top skin cap
# and exposes the three webs standing on the bottom skin -- the conforming junctions
cc = qn[hexes].mean(1)
keep = np.where(cc[:, 1] < 0.03)[0]
solid_cut = grid.extract_cells(keep)
p.add_mesh(solid_cut, color="#e07b39", show_edges=True, edge_color="#3a3a3a", line_width=0.4)
cam(p, zc)
fn = os.path.join(OUTD, "ell_boundary_mesh.png")
p.screenshot(fn, transparent_background=False); p.close(); print("wrote", fn)
