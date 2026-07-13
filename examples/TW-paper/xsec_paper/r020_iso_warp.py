"""IEA r=0.2 out-of-plane axial warping u1 on the DISTORTED mesh, isometric 3-D:
VABS (.U) vs OpenSG RM, side by side.  The section (y2,y3) is displaced out-of-plane by u1."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm

D2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "data", "2d_yaml"))
SHELL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "TW-paper",
                                     "iea22_blade", "data", "shell_r020.yaml"))
FIG = os.path.join(HERE, "figures")
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
U = np.loadtxt(os.path.join(D2, "iea_r020.sg.U"))            # id y2 y3 u1 u2 u3
xy = U[:, 1:3]; u1v = U[:, 3]
step = max(1, len(xy) // 5000)
pts = xy[::step]; uv = u1v[::step]
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")
ur = np.asarray(dehom_rm.disp_at_points(B, pts, beam_force_vabs=FF))[:, 0]   # RM u1

ext = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
um = max(np.abs(uv).max(), np.abs(ur).max())
sc = 0.35 * ext / um                                        # z exaggeration for the warping
vmin, vmax = 1e3 * min(uv.min(), ur.min()), 1e3 * max(uv.max(), ur.max())

fig = plt.figure(figsize=(15, 6.5))
for c, (u, tag) in enumerate([(uv, "VABS (solid)"), (ur, "OpenSG RM (shell dehom)")]):
    ax = fig.add_subplot(1, 2, c + 1, projection="3d")
    s = ax.scatter(pts[:, 0], pts[:, 1], u * sc, c=u * 1e3, cmap="rainbow",
                   vmin=vmin, vmax=vmax, s=4, linewidths=0)
    ax.set_title(r"$u_1$ warping -- %s" % tag, fontsize=11)
    ax.set_xlabel("y2 (m)"); ax.set_ylabel("y3 (m)"); ax.set_zlabel(r"$u_1$ ($\times$%.0f)" % sc)
    ax.view_init(elev=22, azim=-62); ax.set_box_aspect((ext, np.ptp(pts[:, 1]), 0.6 * ext))
    fig.colorbar(s, ax=ax, shrink=0.55, pad=0.02, label=r"$u_1$ (mm)")
fig.suptitle("IEA r=0.2 axial warping $u_1$ on the distorted mesh (isometric)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r020_iso_warp.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote r020_iso_warp.png (scale x%.0f, %d pts)" % (sc, len(pts)))
