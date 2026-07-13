"""IEA r=0.2: (A) VABS .U displacement u1,u2,u3 along the circumferential + spar-cap paths;
(B) VABS vs OpenSG-RM in-plane stress CONTOURS (S11,S22,S12) side by side, rainbow."""
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
FIG = os.path.join(HERE, "figures")
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
U = np.loadtxt(os.path.join(D2, "iea_r020.sg.U"))            # id y2 y3 u1 u2 u3
d = np.loadtxt(os.path.join(D2, "iea_r020.sg.SM"), skiprows=2)
sm_xy, sm_s = d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]     # [S11,S22,S33,S23,S13,S12]
utree = cKDTree(U[:, 1:3]); stree = cKDTree(sm_xy)
cap = np.loadtxt(os.path.join(D2, "solid.lp_sparcap_center_thickness_r020.coords"))[:, :2]
circ = np.loadtxt(os.path.join(D2, "solid.circumferential_r020.coords"))[:, :2]
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")

# ---------------- (A) u1,u2,u3 along the two paths: RM vs VABS overlaid ----------------
def arclen(p): return np.r_[0.0, np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))]
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for r, (P, name, xl) in enumerate([(circ, "circumferential (LP, LE$\\to$TE)", "arc length (m)"),
                                    (cap, "spar-cap through-thickness", "OML$\\to$IML (mm)")]):
    Uv = U[utree.query(P)[1], 3:6] * 1e3                     # VABS .U (mm)
    Ur = np.asarray(dehom_rm.disp_at_points(B, P, beam_force_vabs=FF)) * 1e3   # RM warping (mm)
    x = arclen(P) * (1.0 if r == 0 else 1e3)
    for k, comp in enumerate(["u_1", "u_2", "u_3"]):
        a = ax[r, k]
        a.plot(x, Ur[:, k], "r--o", ms=3, lw=1.6, label="OpenSG RM")
        a.plot(x, Uv[:, k], "g-^", ms=3, lw=1.6, label="VABS ($.\\mathrm{U}$)")
        a.set_title(r"$%s$  --  %s" % (comp, name), fontsize=9)
        a.set_xlabel(xl); a.set_ylabel(r"$%s$ (mm)" % comp); a.grid(True, ls=":", alpha=0.6)
        a.legend(fontsize=7.5)
fig.suptitle("IEA r=0.2 local displacement along the paths: OpenSG RM vs VABS",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(FIG, "r020_disp_paths.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
print("wrote r020_disp_paths.png")

# ---------------- (B) VABS vs RM in-plane stress contours, side by side ----------------
step = max(1, len(sm_xy) // 3000)
pts = sm_xy[::step]; Vsub = sm_s[::step]                     # ~3000 points
Rsub = np.asarray(dehom_rm.stress_at_points(B, pts, beam_force_vabs=FF, frame="material")["stress"])
comps = [(0, "S_{11}"), (1, "S_{22}"), (5, "S_{12}")]
fig, ax = plt.subplots(3, 2, figsize=(12, 10))
for r, (ci, lab) in enumerate(comps):
    v = Vsub[:, ci] / 1e6; rm = Rsub[:, ci] / 1e6
    lo, hi = np.percentile(np.r_[v, rm], [2, 98]); m = max(abs(lo), abs(hi))
    for c, (dat, tag) in enumerate([(v, "VABS (solid)"), (rm, "OpenSG RM (shell dehom)")]):
        a = ax[r, c]
        sc = a.scatter(pts[:, 0], pts[:, 1], c=dat, s=3, cmap="rainbow", vmin=-m, vmax=m, linewidths=0)
        a.set_aspect("equal"); a.axis("off")
        a.set_title(r"$\sigma_{%s}$ -- %s (MPa)" % (lab[2:], tag), fontsize=10)
        if c == 1:
            fig.colorbar(sc, ax=ax[r, :].tolist(), shrink=0.75, pad=0.01)
fig.suptitle("IEA r=0.2 in-plane stress: VABS solid vs OpenSG RM shell dehomogenization",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(FIG, "r020_stress_contour_cmp.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote r020_stress_contour_cmp.png (%d points)" % len(pts))
