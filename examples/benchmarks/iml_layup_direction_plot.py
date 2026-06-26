"""
Diagnostic: IML reference mesh + layup-direction arrows vs the solid mesh.

For station 15, overlay on the 2Dsolid_VABS_15 mesh:
  * OML nodes  (the 1Dshell reference, outer mold line)        - blue
  * IML nodes  (offset inward along the material e3)           - red
  * arrows IML -> OML  (the laminate, length = local thickness, pointing in the
                        layup stacking direction OML<-IML)     - green

If the IML offset and the material e3 are correct, each arrow spans exactly the
solid wall: its tail (red) sits on the solid INNER surface, its head (blue) on
the OUTER surface.  Where an arrow over/under-shoots the wall (e.g. a web or the
TE) the reference is mis-placed there, which is what drives the EA / inertia
differences.
"""
import os
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
from fe_jax import load_yaml, read_mesh, element_e3_from_yaml, offset_oml_to_iml

SHELL15 = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
SOLID15 = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
           r"\training data\opensg-FEniCS\data\2Dsolid_VABS_15.yaml")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _row(r):
    if isinstance(r, str):
        return r.strip("[]").split()
    if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
        return r[0].strip("[]").split()
    return [str(v) for v in r]


def solid_edges(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    vN = np.array([[float(v) for v in _row(n)] for n in d["nodes"]])[:, :2]
    quads = [[int(v) - 1 for v in _row(e)] for e in d["elements"]]
    seen = set(); segs = []
    for q in quads:
        k = len(q)
        for i in range(k):
            a, b = q[i], q[(i + 1) % k]
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key); segs.append([vN[a], vN[b]])
    return vN, segs


# OML + IML shell meshes
nodes3, elements, mat, lay, e2l = load_yaml(SHELL15)
oml, cells, lpe = read_mesh(nodes3, elements, e2l)
e3 = element_e3_from_yaml(SHELL15)
iml = offset_oml_to_iml(oml, cells, lpe, lay, elem_e3=e3)

# solid mesh
vN, segs = solid_edges(SOLID15)
print(f"shell OML nodes: {len(oml)}  |  solid nodes: {len(vN)}, edges: {len(segs)}")

fig, ax = plt.subplots(figsize=(15, 11))
ax.add_collection(LineCollection(segs, colors="0.75", linewidths=0.4, zorder=1))
# arrows IML(tail) -> OML(head)
d = oml - iml
ax.quiver(iml[:, 0], iml[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy",
          scale=1, color="green", width=0.0016, headwidth=4, zorder=3,
          label="laminate (IML->OML, = thickness)")
ax.plot(oml[:, 0], oml[:, 1], "b.", ms=4, zorder=4, label="OML (1Dshell ref)")
ax.plot(iml[:, 0], iml[:, 1], "r.", ms=4, zorder=4, label="IML (offset)")
ax.set_aspect("equal"); ax.autoscale()
ax.set_title("Station 15: IML reference + layup-direction arrows vs solid mesh "
             "(2Dsolid_VABS_15)", fontsize=13, fontweight="bold")
ax.set_xlabel("y2"); ax.set_ylabel("y3"); ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
p = os.path.join(OUT, "iml_layup_direction.png")
os.makedirs(OUT, exist_ok=True); fig.savefig(p, dpi=160); print("wrote", p)
