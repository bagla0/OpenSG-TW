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
cap = np.loadtxt(os.path.join(D2, "solid.lp_sparcap_right_thickness_r020.coords"))[:, :2]
circ = np.loadtxt(os.path.join(D2, "solid.circumferential_r020.coords"))[:, :2]
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")


def arclen(p):
    return np.r_[0.0, np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))]


# ---------------- (1) full-section contour: VABS vs RM, u1,u2,u3 ----------------
# NOTE: the full-section displacement CONTOURS (r020_disp_contour_cmp.png etc.) are produced by
# r020_contours.py on the real solid-mesh triangulation.  This script owns only the path plots.

# ---------------- u1,u2,u3 along each path, INDIVIDUAL files (non-dim x-axis 0..1) ----------------
def path_disp(P, kind, tag):
    Uv = U[utree.query(P)[1], 3:6] * 1e3                     # VABS .U (mm)
    Ur = np.asarray(dehom_rm.disp_at_points(B, P, beam_force_vabs=FF)) * 1e3   # RM warping + director (mm)
    col = 0 if kind == "circ" else 1                         # circ: y2 (LE->TE) ; cap: y3 (OML->IML)
    x = (P[:, col] - P[0, col]) / (P[-1, col] - P[0, col])   # normalized 0..1
    for k, comp in enumerate(["u_1", "u_2", "u_3"]):         # ONE figure per displacement component
        fig, a = plt.subplots(figsize=(6.2, 4.2))
        a.plot(x, Ur[:, k], "r--o", ms=4, lw=1.7, label="OpenSG RM")
        a.plot(x, Uv[:, k], "g-^", ms=4, lw=1.7, label="VABS ($.\\mathrm{U}$)")
        a.set_xlabel("non-dim parameter"); a.set_ylabel(r"$%s$ (mm)" % comp)
        a.set_xlim(-0.03, 1.03); a.grid(True, ls=":", alpha=0.6)
        a.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)  # outside
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "r020_disp_%s_%s.png" % (tag, comp)), dpi=160, bbox_inches="tight")
        plt.close(fig)
    print("wrote r020_disp_%s_u*.png (3 individual)" % tag)


path_disp(circ, "circ", "circ")
path_disp(cap, "cap", "cap")
