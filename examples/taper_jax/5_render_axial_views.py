"""5_render_axial_views.py -- PNGs of the ACTUAL generated r=0.2->0.3 meshes viewed down
the BEAM AXIS (beam axis out of the plane of view, cross-section face-on):

  iea_hybrid_axial.png      : SOLID mixed segment (hex skin + tet webs), by material
  iea_hybrid_boundary_L.png : the extracted L boundary (MIXED quad4 skin + tri3 webs)
  iea_shell_axial.png       : QUAD shell taper segment (ring-lofted), by section set

Renders the real computed cells with PyVista (software GL), never a sketch.

    python 5_render_axial_views.py
"""
import os
import sys

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "taper"))
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))

import pyvista as pv
from matplotlib.colors import ListedColormap
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import hex_between_sections, shell_between_sections, solid_yaml_payload
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, extract_boundary_submesh,
                                           _PERM3, _PERMF)

PAL = np.array([[0.42, 0.42, 0.42], [0.12, 0.47, 0.71], [0.17, 0.63, 0.17], [1.00, 0.50, 0.05],
                [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.09, 0.75, 0.81], [0.84, 0.15, 0.16]])

print("building meshes ...", flush=True)
blade = load_blade(WINDIO)
cs1 = build_cross_section(blade, 0.2, mesh_size=0.02)
cs2 = build_cross_section(blade, 0.3, mesh_size=0.02)
z1, z2 = blade_span_z(blade, 0.2), blade_span_z(blade, 0.3)
res = hex_between_sections(cs1, cs2, z1, z2, nr=4, nsp=12, nw=3, mesh_size=0.02)
oris, hmats = solid_yaml_payload(res, cs1, cs2)
web_mask = np.array([t[0] == "web" for t in res["htag"]])
mat_names = sorted(set(hmats))
name_ix = {n: i for i, n in enumerate(mat_names)}
seg = dict(nodes=np.asarray(res["nodes"])[:, _PERM3],
           batches={"hex8": (np.asarray(res["hexes"]),
                             np.array([name_ix[m] for m in hmats], int),
                             np.asarray(oris)[:, _PERMF])},
           mat_param=np.zeros((len(mat_names), 9)), nelem=len(res["hexes"]))
hyb = split_batches_to_tets(seg, mask=web_mask)


def to_grid(nodes_bf, batches, data_of):
    """UnstructuredGrid from beam-first nodes plotted as (x2, x3, beam) so pl.view_xy()
    looks DOWN THE BEAM AXIS (beam axis out of the plane of view)."""
    P = np.asarray(nodes_bf)[:, [1, 2, 0]]
    cells, ct, cd = [], [], []
    VTK = {8: pv.CellType.HEXAHEDRON, 4: pv.CellType.TETRA}
    V2D = {4: pv.CellType.QUAD, 3: pv.CellType.TRIANGLE}
    for kind, (conn, mid, _f) in batches.items():
        n = conn.shape[1]
        t = VTK[n] if kind in ("hex8", "tet4") else V2D[n]
        for row, m in zip(conn, mid):
            cells.append(np.r_[n, row])
            ct.append(t)
            cd.append(data_of(m))
    grid = pv.UnstructuredGrid(np.concatenate(cells), np.array(ct, np.uint8), P)
    grid.cell_data["set"] = np.array(cd, int)
    return grid


def shot(grid, png, title, nset):
    pl = pv.Plotter(off_screen=True, window_size=(1500, 700))
    pl.add_mesh(grid, scalars="set", cmap=ListedColormap(PAL[np.arange(max(nset, 1)) % len(PAL)]),
                show_edges=True, edge_color="black", line_width=0.35, show_scalar_bar=False)
    pl.add_text(title + "   (beam axis OUT of plane)", font_size=11)
    pl.view_xy()                                       # camera along +beam -> section face-on
    pl.camera.zoom(1.35)
    pl.screenshot(os.path.join(HERE, png))
    pl.close()
    print("wrote", png, flush=True)


# 1. hybrid solid segment, axial view, by material
g = to_grid(hyb["nodes"], hyb["batches"], lambda m: m)
shot(g, "iea_hybrid_axial.png",
     "SOLID mixed hex+tet segment r=0.2-0.3, by material", len(mat_names))

# 2. extracted L boundary (mixed quad skin + tri webs), face-on
bL = extract_boundary_submesh(hyb, "L")
gb = to_grid(bL["nodes"], bL["batches"], lambda m: m)
shot(gb, "iea_hybrid_boundary_L.png",
     "L boundary: MIXED quad4 skin + tri3 webs, by material", len(mat_names))

# 3. quad shell segment, axial view, by section set
shell = shell_between_sections(res, cs1, cs2, reference="OML")
regs = sorted(set(shell["region_of_quad"]))
rix = {r: i for i, r in enumerate(regs)}
sn = np.asarray(shell["nodes"])                        # (x, y, z=beam) -> view down z
gq = pv.UnstructuredGrid(
    np.concatenate([np.r_[4, q] for q in np.asarray(shell["quads"], int)]),
    np.full(len(shell["quads"]), pv.CellType.QUAD, np.uint8), sn)
gq.cell_data["set"] = np.array([rix[r] for r in shell["region_of_quad"]], int)
shot(gq, "iea_shell_axial.png",
     "QUAD shell taper segment (ring-lofted, OML), by region", len(regs))
