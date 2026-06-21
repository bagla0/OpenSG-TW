"""Debug the IML: show the IML-OFFSET geometry (what the solver actually homogenizes at frac=1.0) with
its e1/e2/e3.  e1=(0,0,1) is out-of-plane; e2 = offset element tangent, e3 = offset inward normal.
Grey = OML contour, navy = IML-offset contour.  A rear-web junction zoom reveals any residual fold.
f=0.2 (thin, should be clean) vs f=0.6 (thick, where IML is worst)."""
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


def get_iml(fi):
    shy = os.path.join(HERE, "shell_ref_f%03d_connect.yaml" % fi)
    n3d, elements, mat_db, layup_db, e2l = load_yaml(shy)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    nodes = np.asarray(nodes, float)[:, :2]
    e3 = np.asarray(element_e3_from_yaml(shy), float)
    iml = np.asarray(offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=1.0), float)
    return nodes, cells, iml


def draw(ax, nodes, cells, iml, zoom=None):
    ends = cells[:, [0, -1]]
    for (a, b) in ends:
        ax.plot(nodes[[a, b], 0], nodes[[a, b], 1], color="0.75", lw=0.6, zorder=1)
        ax.plot(iml[[a, b], 0], iml[[a, b], 1], color="navy", lw=1.0, zorder=2)
    P0, P1 = iml[ends[:, 0]], iml[ends[:, 1]]; ctr = 0.5 * (P0 + P1); t = P1 - P0
    Ln = np.linalg.norm(t, axis=1, keepdims=True); e2 = t / (Ln + 1e-30)
    e3v = np.stack([-e2[:, 1], e2[:, 0]], 1)
    cen = iml.mean(0); fl = ((cen - ctr) * e3v).sum(1) < 0; e3v[fl] = -e3v[fl]
    al = 0.03
    ax.quiver(ctr[:, 0], ctr[:, 1], e2[:, 0], e2[:, 1], color="tab:blue", angles="xy",
              scale_units="xy", scale=1 / al, width=0.004, zorder=3, label="e2 tangent")
    ax.quiver(ctr[:, 0], ctr[:, 1], e3v[:, 0], e3v[:, 1], color="tab:red", angles="xy",
              scale_units="xy", scale=1 / al, width=0.004, zorder=3, label="e3 normal")
    ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    if zoom:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
    # flag folded elements (IML tangent reversed vs OML tangent)
    ot = nodes[ends[:, 1]] - nodes[ends[:, 0]]
    rev = (ot * t).sum(1) < 0
    if rev.any():
        ax.plot(ctr[rev, 0], ctr[rev, 1], "x", color="magenta", ms=9, mew=2, zorder=5,
                label="FOLDED (%d)" % rev.sum())


fig, axs = plt.subplots(2, 2, figsize=(16, 9))
for row, fi in enumerate([20, 60]):
    nodes, cells, iml = get_iml(fi)
    draw(axs[row, 0], nodes, cells, iml)
    axs[row, 0].set_title("f=0.%d  IML offset (grey=OML, navy=IML),  e1=(0,0,1) out-of-plane" % (fi // 10), fontsize=11)
    axs[row, 0].legend(fontsize=8, loc="upper right")
    draw(axs[row, 1], nodes, cells, iml, zoom=(0.38, 0.62, -0.13, 0.13))
    axs[row, 1].set_title("f=0.%d  rear-web junction zoom" % (fi // 10), fontsize=11)
fig.suptitle("IML-offset geometry & e1/e2/e3 (magenta x = folded element)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "iml_orient_debug.png"), dpi=150, bbox_inches="tight")
print("wrote figures/iml_orient_debug.png")
