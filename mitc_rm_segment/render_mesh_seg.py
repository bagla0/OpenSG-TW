"""render_mesh_seg.py <seg.yaml> <out.png> [iso|end] -- render a lofted segment mesh
(8-node hex solid or 4-node quad shell) with PyVista.  iso = full isometric (beam axis
horizontal); end = look down the beam axis to show the through-thickness layers.
"""
import sys
import numpy as np
import yaml
import pyvista as pv

pv.OFF_SCREEN = True
CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
SRC, OUT = sys.argv[1], sys.argv[2]
VIEW = sys.argv[3] if len(sys.argv) > 3 else "iso"


def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


d = yaml.load(open(SRC), Loader=CL)
nodes = np.array([_row(r)[:3] for r in d["nodes"]])
elems = [[int(v) - 1 for v in _row(e)] for e in d["elements"]]
nen = len(elems[0])
ctype = {8: pv.CellType.HEXAHEDRON, 4: pv.CellType.QUAD, 6: pv.CellType.WEDGE}[nen]
# beam axis (z, 3rd coord) -> world x so the segment lies horizontal
P = np.column_stack([nodes[:, 2], nodes[:, 0], nodes[:, 1]])
cells = np.hstack([[nen] + e for e in elems])
grid = pv.UnstructuredGrid(cells, np.full(len(elems), ctype, np.uint8), P)
print("%d nodes, %d %s cells" % (len(nodes), len(elems),
      {8: "hex", 4: "quad", 6: "wedge"}[nen]))

col = "#e0954a" if nen != 4 else "#c98a3a"
p = pv.Plotter(off_screen=True, window_size=(1200, 780), border=False, lighting="three lights")
p.add_mesh(grid, color=col, show_edges=True, edge_color="#3a3a3a", line_width=0.4,
           ambient=0.35, diffuse=0.75)
p.enable_parallel_projection()
if VIEW == "end":
    p.camera_position = "yz"                              # look down beam axis
    p.reset_camera(); p.camera.zoom(1.5)
else:
    p.view_isometric(); p.camera.azimuth = -18; p.camera.elevation = 12
    p.reset_camera(); p.camera.zoom(1.25)
p.screenshot(OUT); p.close(); print("wrote", OUT)
