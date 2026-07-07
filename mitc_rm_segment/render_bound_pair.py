"""render_bound_pair.py <solid_2d.yaml> <shell_1d.yaml> <out.png> -- side-by-side
boundary meshes for the webbed ellipse: 2-D solid cross-section (filled triangles,
left) and 1-D mid-surface shell contour (skin + webs as lines, right).  Actual meshes
rendered from the YAMLs with PyVista.
"""
import sys
import numpy as np
import yaml
import pyvista as pv

pv.OFF_SCREEN = True
CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
SOL, SHL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


# ---- solid 2D triangles ----
ds = yaml.load(open(SOL), Loader=CL)
sx = np.array([_row(r)[:2] for r in ds["nodes"]])
st = [[int(v) - 1 for v in _row(e)] for e in ds["elements"]]
sx = sx - sx.mean(0)
faces = np.hstack([np.hstack([[3], t]) for t in st])
gsolid = pv.PolyData(np.column_stack([sx[:, 0], sx[:, 1], np.zeros(len(sx))]), faces)

# ---- shell 1D contour, skin vs web ----
dh = yaml.load(open(SHL), Loader=CL)
hx = np.array([_row(r)[:2] for r in dh["nodes"]])
he = [[int(v) - 1 for v in _row(e)] for e in dh["elements"]]
hx = hx - hx.mean(0)
web_names = {g["name"] for g in dh["sets"]["element"] if "1" in g["name"] and g["name"] != "layup_0"}
lab_web = set()
secs = dh.get("sections", [])
# section 0 = skin (layup_0), others = web
web_set_ids = [i for i, s in enumerate(secs) if s.get("elementSet") != "layup_0"]
name_web = {g["name"] for g in dh["sets"]["element"] if g["name"] != "layup_0"}
elem_isweb = np.zeros(len(he), dtype=bool)
for g in dh["sets"]["element"]:
    if g["name"] in name_web:
        for lab in g["labels"]:
            elem_isweb[int(lab) - 1] = True


def lines_poly(elems):
    segs = np.hstack([[2, e[0], e[1]] for e in elems])
    return pv.PolyData(np.column_stack([hx[:, 0], hx[:, 1], np.zeros(len(hx))]), lines=segs)


skin_lines = lines_poly([e for e, w in zip(he, elem_isweb) if not w])
web_lines = lines_poly([e for e, w in zip(he, elem_isweb) if w])

p = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1500, 620), border=False)
p.subplot(0, 0)
p.add_text("solid boundary (2-D)", font_size=12, color="#333333")
p.add_mesh(gsolid, color="#e9b98a", show_edges=True, edge_color="#7a4a1e", line_width=0.7)
p.enable_parallel_projection(); p.camera_position = "xy"; p.reset_camera(); p.camera.zoom(1.25)
p.subplot(0, 1)
p.add_text("shell boundary (1-D mid-surface)", font_size=12, color="#333333")
p.add_mesh(skin_lines, color="#b5651d", line_width=3, render_lines_as_tubes=True)
p.add_mesh(web_lines, color="#1f6f8b", line_width=3, render_lines_as_tubes=True)
p.enable_parallel_projection(); p.camera_position = "xy"; p.reset_camera(); p.camera.zoom(1.25)
p.screenshot(OUT); p.close()
print("solid: %d nodes/%d tris | shell: %d nodes/%d lines (%d web)"
      % (len(sx), len(st), len(hx), len(he), int(elem_isweb.sum())))
print("wrote", OUT)
