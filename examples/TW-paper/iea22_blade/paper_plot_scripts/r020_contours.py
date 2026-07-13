"""r020_contours.py -- IEA r=0.2 local-field contours rendered on the REAL solid mesh
triangulation (matplotlib tricontourf, not scattered points), VABS solid vs OpenSG RM.

Produces, with NO "IEA r=0.2" super-title and a rainbow scheme:
  figures/r020_disp_contour_cmp.png     u1,u2,u3   VABS | RM   (per-row shared colorbar)
  figures/r020_stress_contour_cmp.png   S11,S22,S12 VABS | RM  (per-row shared colorbar)
  figures/r020_disp_mag.png             |u|=sqrt(u1^2+u2^2+u3^2) VABS | RM, ONE shared
                                        colorbar, no label under it
Individual panels (VABS alone / OpenSG alone), stored beside for the data folder:
  figures/r020_disp_contour_vabs.png  r020_disp_contour_rm.png
  figures/r020_stress_contour_vabs.png r020_stress_contour_rm.png
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm

D2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "data", "2d_yaml"))
VABS = os.path.join(D2, "IEA_VABS")
SHELL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "TW-paper",
                                     "iea22_blade", "data", "shell_r020.yaml"))
SOLID = os.path.join(D2, "iea_r020_solid.yaml")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
CMAP = "rainbow"


def _row(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


# ---- solid mesh triangulation (nodes + tris) ----
ds = yaml.safe_load(open(SOLID))
xy_yaml = np.array([_row(n)[:2] for n in ds["nodes"]])
tris = np.array([[int(round(x)) - 1 for x in _row(e)] for e in ds["elements"]], dtype=int)
tris = tris[:, :3]                                   # all-tri mesh

# ---- VABS fields at nodes ----
U = np.loadtxt(os.path.join(VABS, "iea_r020.sg.U"))          # id y2 y3 u1 u2 u3
U = U[np.argsort(U[:, 0])]
xy = U[:, 1:3]; uV = U[:, 3:6] * 1e3                          # node disp (mm)
d = np.loadtxt(os.path.join(VABS, "iea_r020.sg.SM"), skiprows=2)
sm_xy, sm_s = d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]      # [S11,S22,S33,S23,S13,S12]
# VABS stress interpolated to nodes (linear, fill boundary gaps with nearest)
sV = griddata(sm_xy, sm_s[:, [0, 1, 5]], xy, method="linear")
nn = cKDTree(sm_xy).query(xy)[1]
bad = ~np.isfinite(sV).all(1); sV[bad] = sm_s[nn][bad][:, [0, 1, 5]]
sV = sV / 1e6                                                 # MPa  [S11,S22,S12]

# the solid YAML `tris` index node ids 1..N; VABS .U (sorted by id) must share that ordering, so
# .U row i == YAML node id i+1.  Guard it: a mismatch would silently scramble the triangulation.
assert xy.shape == xy_yaml.shape and np.allclose(xy, xy_yaml, atol=1e-3), \
    "VABS .U node order != solid-YAML node order -- triangulation would be scrambled"
tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)

# ---- RM fields at the SAME nodes ----
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")
uR = np.asarray(dehom_rm.disp_at_points(B, xy, beam_force_vabs=FF)) * 1e3               # (N,3) mm
sR = np.asarray(dehom_rm.stress_at_points(B, xy, beam_force_vabs=FF, frame="material")["stress"])
sR = sR[:, [0, 1, 5]] / 1e6                                                             # (N,3) MPa

NL = 24


def tcf(ax, val, vmin, vmax, sym=True):
    # tripcolor on the ACTUAL mesh cells (Gouraud) -- every triangle is rendered faithfully, so web
    # junctions and thin structure are correct (no tricontourf cross-cell contouring artifacts); this
    # is the same nodal Gouraud shading ParaView/VTK produce from the exported .vtk.
    cs = ax.tripcolor(tri, np.clip(val, vmin, vmax), shading="gouraud", cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.axis("off")
    return cs


def comp_cmp(V, R, comps, out, mag_symbol):
    """3 rows x 2 cols (VABS|RM), per-row shared symmetric scale, one colorbar per row."""
    fig, ax = plt.subplots(len(comps), 2, figsize=(11.5, 3.1 * len(comps)))
    if len(comps) == 1:
        ax = ax[None, :]
    for r, (ci, lab) in enumerate(comps):
        m = np.nanpercentile(np.abs(np.r_[V[:, ci], R[:, ci]]), 99) or 1e-9
        cs = None
        for c, (dat, tag) in enumerate([(V[:, ci], "VABS (solid)"), (R[:, ci], "OpenSG RM")]):
            cs = tcf(ax[r, c], dat[:], -m, m)
            ax[r, c].set_title(r"$%s_{%s}$ -- %s" % (mag_symbol, lab, tag), fontsize=10.5)
        cb = fig.colorbar(cs, ax=ax[r, :].tolist(), shrink=0.82, pad=0.015)
        cb.set_label("MPa" if mag_symbol == "\\sigma" else "mm", fontsize=9)
    fig.savefig(out, dpi=155, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))


def comp_single(F, comps, out, mag_symbol, who):
    """1 row x len(comps): one solver, its components, individual colorbars."""
    fig, ax = plt.subplots(1, len(comps), figsize=(4.6 * len(comps), 3.4))
    for k, (ci, lab) in enumerate(comps):
        m = np.nanpercentile(np.abs(F[:, ci]), 99) or 1e-9
        cs = tcf(ax[k], F[:, ci], -m, m)
        ax[k].set_title(r"$%s_{%s}$ -- %s" % (mag_symbol, lab, who), fontsize=10.5)
        fig.colorbar(cs, ax=ax[k], shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=155, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))


DISP = [(0, "1"), (1, "2"), (2, "3")]
STRS = [(0, "11"), (1, "22"), (2, "12")]

# ---- comparison contours (paper) ----
comp_cmp(uV, uR, DISP, os.path.join(FIG, "r020_disp_contour_cmp.png"), "u")
comp_cmp(sV, sR, STRS, os.path.join(FIG, "r020_stress_contour_cmp.png"), "\\sigma")

# ---- individual panels (data folder) ----
comp_single(uV, DISP, os.path.join(FIG, "r020_disp_contour_vabs.png"), "u", "VABS (solid)")
comp_single(uR, DISP, os.path.join(FIG, "r020_disp_contour_rm.png"), "u", "OpenSG RM")
comp_single(sV, STRS, os.path.join(FIG, "r020_stress_contour_vabs.png"), "\\sigma", "VABS (solid)")
comp_single(sR, STRS, os.path.join(FIG, "r020_stress_contour_rm.png"), "\\sigma", "OpenSG RM")

# ---- displacement MAGNITUDE |u| on the DISTORTED shape (deform by in-plane u2,u3), VABS | RM,
#      ONE shared colorbar (no label, no extend triangle) ----
magV = np.linalg.norm(uV, axis=1); magR = np.linalg.norm(uR, axis=1)
vmax = np.nanpercentile(np.r_[magV, magR], 99.5)
ipV = uV[:, 1:3] * 1e-3; ipR = uR[:, 1:3] * 1e-3                    # in-plane (u2,u3) in m
ext = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
sc = 0.11 * ext / max(np.linalg.norm(np.r_[ipV, ipR], axis=1).max(), 1e-12)   # visual exaggeration
lv = np.linspace(0, vmax, NL)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
cs = None
for a, (dat, ip, tag) in zip(ax, [(magV, ipV, "VABS (solid)"), (magR, ipR, "OpenSG RM")]):
    dxy = xy + ip * sc                                             # distorted section
    dtri = mtri.Triangulation(dxy[:, 0], dxy[:, 1], tris)
    cs = a.tripcolor(dtri, np.clip(dat, 0, vmax), shading="gouraud", cmap=CMAP, vmin=0, vmax=vmax)
    a.set_aspect("equal"); a.axis("off")
    a.set_title(r"$|u|$ -- %s" % tag, fontsize=11)
cb = fig.colorbar(cs, ax=ax.tolist(), shrink=0.85, pad=0.02)       # one shared bar, no label, no triangle
fig.savefig(os.path.join(FIG, "r020_disp_mag.png"), dpi=155, bbox_inches="tight"); plt.close(fig)
print("wrote r020_disp_mag.png  (distort x%.0f, |u| max V=%.3f RM=%.3f mm)" % (sc, magV.max(), magR.max()))

# ---- export the solid mesh + all dehom fields as .vtk (one file per contour) for ParaView ----
try:
    import pyvista as pv
    VTK = os.path.join(HERE, "vtk"); os.makedirs(VTK, exist_ok=True)
    pts3 = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).astype(np.int64).ravel()
    base = pv.PolyData(pts3, faces)
    fields = {"S11": (sV[:, 0], sR[:, 0]), "S22": (sV[:, 1], sR[:, 1]), "S12": (sV[:, 2], sR[:, 2]),
              "u1": (uV[:, 0], uR[:, 0]), "u2": (uV[:, 1], uR[:, 1]), "u3": (uV[:, 2], uR[:, 2]),
              "umag": (magV, magR)}
    for name, (fv, fr) in fields.items():                          # one .vtk per contour (VABS + RM arrays)
        m = base.copy()
        m.point_data["VABS_" + name] = np.asarray(fv)
        m.point_data["OpenSG_RM_" + name] = np.asarray(fr)
        m.save(os.path.join(VTK, "r020_%s.vtk" % name))
    print("wrote %d .vtk contour files -> %s (open in ParaView, colour by VABS_/OpenSG_RM_)" %
          (len(fields), VTK))
except Exception as e:
    print("vtk export skipped:", repr(e)[:100])
