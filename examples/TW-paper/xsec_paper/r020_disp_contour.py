"""IEA r=0.2 local DISPLACEMENT recovery along the paths, OpenSG-RM vs VABS:
  figures/r020_disp_circ.png  -- u1,u2,u3 along the LP circumferential path (individual);
  figures/r020_disp_cap.png   -- u1,u2,u3 along the LEFT spar-cap column (individual).
(The full-section displacement CONTOURS are produced by r020_contours.py on the real solid
mesh triangulation.)  Same beam load FF as the stress recovery.  RM warping =
dehom_rm.disp_at_points (leading-order mid-surface); VABS = the .U warping field nearest each node."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm

D2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "data", "2d_yaml"))
SHELL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "TW-paper",
                                     "iea22_blade", "data", "shell_r020.yaml"))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
U = np.loadtxt(os.path.join(D2, "iea_r020.sg.U"))            # id y2 y3 u1 u2 u3
utree = cKDTree(U[:, 1:3])
cap = np.loadtxt(os.path.join(D2, "solid.lp_sparcap_left_thickness_r020.coords"))[:, :2]
circ = np.loadtxt(os.path.join(D2, "solid.circumferential_r020.coords"))[:, :2]
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")


def arclen(p):
    return np.r_[0.0, np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))]


# ---------------- (1) full-section contour: VABS vs RM, u1,u2,u3 ----------------
# NOTE: the full-section displacement CONTOURS (r020_disp_contour_cmp.png etc.) are produced by
# r020_contours.py on the real solid-mesh triangulation.  This script owns only the path plots.

# ---------------- u1,u2,u3 along each path, INDIVIDUAL files ----------------
def path_disp(P, name, xl, xscale, out):
    Uv = U[utree.query(P)[1], 3:6] * 1e3                     # VABS .U (mm)
    Ur = np.asarray(dehom_rm.disp_at_points(B, P, beam_force_vabs=FF)) * 1e3   # RM warping (mm)
    x = arclen(P) * xscale
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    for k, comp in enumerate(["u_1", "u_2", "u_3"]):
        a = ax[k]
        a.plot(x, Ur[:, k], "r--o", ms=3, lw=1.6, label="OpenSG RM")
        a.plot(x, Uv[:, k], "g-^", ms=3, lw=1.6, label="VABS ($.\\mathrm{U}$)")
        a.set_title(r"$%s$" % comp, fontsize=11)
        a.set_xlabel(xl); a.set_ylabel(r"$%s$ (mm)" % comp); a.grid(True, ls=":", alpha=0.6)
        a.legend(fontsize=8)
    fig.suptitle("Local displacement along the %s: OpenSG RM vs VABS" % name,
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))


path_disp(circ, "circumferential path (LP, LE$\\to$TE)", "arc length (m)", 1.0,
          os.path.join(FIG, "r020_disp_circ.png"))
path_disp(cap, "left spar-cap through-thickness", "OML$\\to$IML (mm)", 1e3,
          os.path.join(FIG, "r020_disp_cap.png"))
