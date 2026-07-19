"""render_blade_modes.py -- blade first buckling mode: JAX-FEA vs RM-OpenSG, and the span-N field.
Colors the (undeformed) blade skin by |mode displacement| to show WHERE the local buckle localizes."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
m = np.load(os.path.join(D, "blade_mesh.npz")); nodes = m["nodes"]; quads = m["quads"]


def panel(ax, scal, cmap, title, vmax=None):
    verts = nodes[quads]                                     # (ne,4,3)
    cvals = scal[quads].mean(1)
    vmax = vmax or np.percentile(np.abs(cvals), 99)
    pc = Poly3DCollection(verts, linewidths=0)
    pc.set_array(cvals); pc.set_cmap(cmap); pc.set_clim(0, vmax)
    ax.add_collection3d(pc)
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax.set_ylim(-8, 8); ax.set_zlim(-8, 8)
    ax.set_box_aspect((6, 1, 1)); ax.view_init(elev=22, azim=-70); ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    return pc


fig = plt.figure(figsize=(13, 7))
for r, tag in enumerate(["fea", "rm"]):
    fp = os.path.join(D, "blade_%s.npz" % tag)
    if not os.path.exists(fp):
        continue
    z = np.load(fp); modes = z["modes"]; loads = z["loads"]
    umag = np.linalg.norm(modes[:, :3, 0], axis=1)
    ax = fig.add_subplot(2, 1, r + 1, projection="3d")
    panel(ax, umag, "inferno", "%s  mode 1  $\\lambda_1=%.3f$" %
          ("JAX-FEA" if tag == "fea" else "RM-OpenSG", loads[0]))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "blade_modes.png"), dpi=130, bbox_inches="tight")
print("wrote blade_modes.png")

# span-N field (FEA vs RM), undeformed
fig2 = plt.figure(figsize=(13, 7))
for r, tag in enumerate(["fea", "rm"]):
    fp = os.path.join(D, "blade_%s.npz" % tag)
    if not os.path.exists(fp):
        continue
    z = np.load(fp); Nsp = z["Nvec"][:, 0]                  # spanwise membrane N (e1=span)
    verts = nodes[quads]
    ax = fig2.add_subplot(2, 1, r + 1, projection="3d")
    vmax = np.percentile(np.abs(Nsp), 98)
    pc = Poly3DCollection(verts, linewidths=0)
    pc.set_array(Nsp); pc.set_cmap("RdBu_r"); pc.set_clim(-vmax, vmax)
    ax.add_collection3d(pc)
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max()); ax.set_ylim(-8, 8); ax.set_zlim(-8, 8)
    ax.set_box_aspect((6, 1, 1)); ax.view_init(elev=22, azim=-70); ax.set_axis_off()
    ax.set_title("%s  spanwise $N$ (N/m)" % ("JAX-FEA" if tag == "fea" else "RM-OpenSG"), fontsize=10)
fig2.tight_layout(); fig2.savefig(os.path.join(FIG, "blade_N.png"), dpi=130, bbox_inches="tight")
print("wrote blade_N.png")
