"""e1/e2/e3 orientation of the OML (frac=0) and CENTER (frac=0.5) reference surfaces.
e1=(0,0,1) out-of-plane; e2 = element tangent (blue), e3 = inward normal (red).  Grey = OML contour,
navy = reference surface.  OML is f-independent; center is shown for f=0.2 and f=0.6.  magenta x =
folded element (none expected for OML/center -- center offset is half the IML)."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_mesh import load_yaml, read_mesh, offset_oml_to_iml, element_e3_from_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CC, "mh104_thickness_study", "figures")


def get_geom(fi, frac):
    shy = os.path.join(HERE, "shell_ref_f%03d_connect.yaml" % fi)
    n3d, elements, mat_db, layup_db, e2l = load_yaml(shy)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    nodes = np.asarray(nodes, float)[:, :2]
    if frac > 0:
        e3 = np.asarray(element_e3_from_yaml(shy), float)
        g = np.asarray(offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac), float)
    else:
        g = nodes.copy()
    return nodes, cells, g


def draw(ax, oml, cells, geom, zoom=None):
    ends = cells[:, [0, -1]]
    for (a, b) in ends:
        ax.plot(oml[[a, b], 0], oml[[a, b], 1], color="0.75", lw=0.6, zorder=1)
        ax.plot(geom[[a, b], 0], geom[[a, b], 1], color="navy", lw=1.0, zorder=2)
    P0, P1 = geom[ends[:, 0]], geom[ends[:, 1]]; ctr = 0.5 * (P0 + P1); t = P1 - P0
    Ln = np.linalg.norm(t, axis=1, keepdims=True); e2 = t / (Ln + 1e-30)
    e3v = np.stack([-e2[:, 1], e2[:, 0]], 1)
    cen = geom.mean(0); fl = ((cen - ctr) * e3v).sum(1) < 0; e3v[fl] = -e3v[fl]
    al = 0.03
    ax.quiver(ctr[:, 0], ctr[:, 1], e2[:, 0], e2[:, 1], color="tab:blue", angles="xy",
              scale_units="xy", scale=1 / al, width=0.004, zorder=3, label="e2 tangent")
    ax.quiver(ctr[:, 0], ctr[:, 1], e3v[:, 0], e3v[:, 1], color="tab:red", angles="xy",
              scale_units="xy", scale=1 / al, width=0.004, zorder=3, label="e3 normal")
    ot = oml[ends[:, 1]] - oml[ends[:, 0]]; rev = (ot * t).sum(1) < 0
    if rev.any():
        ax.plot(ctr[rev, 0], ctr[rev, 1], "x", color="magenta", ms=9, mew=2, zorder=5, label="folded(%d)" % rev.sum())
    ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    if zoom:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])


specs = [("OML  (frac=0, f-independent)", 20, 0.0),
         ("center  (frac=0.5)  f=0.2", 20, 0.5),
         ("center  (frac=0.5)  f=0.6", 60, 0.5)]
fig, axs = plt.subplots(3, 2, figsize=(15, 13))
for row, (nm, fi, frac) in enumerate(specs):
    oml, cells, geom = get_geom(fi, frac)
    draw(axs[row, 0], oml, cells, geom)
    axs[row, 0].set_title(nm + "   (e1=(0,0,1) out-of-plane)", fontsize=11)
    axs[row, 0].legend(fontsize=8, loc="upper right")
    draw(axs[row, 1], oml, cells, geom, zoom=(0.38, 0.62, -0.13, 0.13))
    axs[row, 1].set_title(nm + " - rear-web zoom", fontsize=11)
fig.suptitle("OML & center reference surfaces: e1/e2/e3  (grey=OML contour, navy=reference)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ref_orient_oml_center.png"), dpi=150, bbox_inches="tight")
print("wrote figures/ref_orient_oml_center.png")
