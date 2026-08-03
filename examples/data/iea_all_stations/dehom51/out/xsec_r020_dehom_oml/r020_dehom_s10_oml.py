"""r020_dehom_s10.py -- r=0.2 (iea_s10) cross-section dehom contours + 3-D deformed shape, RM vs VABS.

CONFORMAL (non-bleeding) stress: the stress is evaluated at the VABS .SM GAUSS points (3 per element,
each INTERIOR to one element = unambiguous material), a linear field is fit through the 3 gauss values
and extrapolated to that element's 3 corners, and everything is rendered on an EXPLODED mesh (each
element keeps its OWN nodes).  So the stress stays DISCONTINUOUS across material (web/skin) boundaries
-- no nodal Gouraud "triangle-weld" bleeding at the web junctions.  Displacement is continuous, so it
stays a normal nodal field.

Fields are LOCAL (warping-level), on the VABS-.sg mesh (which the .SM/.U are native to):
  * stress : RM two-step dehom (material frame) vs VABS .SM, gauss->corner, exploded.
  * disp   : RM warping (disp_at_points) vs VABS WARPING = .U minus the rigid beam disp/rotation
             (u_warp = C^-1 (u - u_g + r) - r ; beam kinematics from the VABS BeamDyn).

Outputs (this folder):
  figures/r020_stress_contour_cmp.png   sigma11/22/12  VABS | RM   (exploded, non-bleeding)
  figures/r020_disp_contour_cmp.png     u1/u2/u3       VABS | RM   (nodal, local warping)
  figures/r020_deformed3d.png           section warped by (u1,u2,u3) in 3-D (out-of-plane u1) VABS | RM
  vtk/r020_dehom_s10.vtk                EXPLODED mesh, VABS_/OpenSG_RM_ {S11,S22,S12,u1,u2,u3,umag}
                                        + disp vectors  -> render 6 field PNGs with render_pv.py
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
sys.path.insert(0, XSEC)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm

SHELL = os.path.join(ROOT, "shell51", "1d_yaml_oml", "iea_s10_shell.yaml")
VABS = os.path.join(IEA, "out", "VABS_iea51")
FF = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))[10, 1:]
BD_VABS = os.path.join(VABS, "iea51vabs_bd_driver.out")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
VTK = os.path.join(HERE, "vtk"); os.makedirs(VTK, exist_ok=True)
CMAP = "rainbow"


def beam_kinematics(path, node):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]]); return u_g, C
    raise ValueError("no BeamDyn header")


def parse_sg_conn(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    hi = next(i for i, l in enumerate(L) if len(l.split()) == 3
              and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
    nn, ne, nm = [int(x) for x in L[hi].split()]
    conn = [[int(x) for x in L[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
    return nn, conn


# ---- mesh = VABS .sg (nodes from .U, tri connectivity from .sg) ----
U = np.loadtxt(os.path.join(VABS, "iea_s10.sg.U")); U = U[np.argsort(U[:, 0])]
xy = U[:, 1:3]; uV_tot = U[:, 3:6]
nn, conn = parse_sg_conn(os.path.join(VABS, "iea_s10.sg"))
assert nn == len(xy)
trl = []
for c in conn:
    c0 = [n - 1 for n in c]
    trl.append(c0[:3]) if len(c0) == 3 else (trl.extend([[c0[0], c0[1], c0[2]], [c0[0], c0[2], c0[3]]]) if len(c0) == 4 else None)
tris = np.array(trl); tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris); M = tris.shape[0]
print("VABS .sg mesh: %d nodes, %d tris" % (len(xy), M), flush=True)

# ---- VABS warping (nodal) ----
u_g, C = beam_kinematics(BD_VABS, 11); Cinv = np.linalg.inv(C)
r3 = np.column_stack([np.zeros(len(xy)), xy[:, 0], xy[:, 1]])
uV = ((Cinv @ (uV_tot - u_g + r3).T).T - r3) * 1e3                    # VABS warping (mm), continuous

# ---- VABS gauss stress (from .SM) ----
dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]; sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6    # VABS gauss Voigt [11,22,33,23,13,12]

# ---- RM warping (nodal) + RM gauss stress -- the expensive RM evaluations, cached for fast re-render ----
_cache = os.path.join(HERE, "_rm_s10_oml_cache.npz")
if os.path.exists(_cache):
    _z = np.load(_cache); uR = _z["uR"]; sRg = _z["sRg"]; print("loaded RM cache", flush=True)
else:
    B = dehom_rm.build_rm_bundle(SHELL)
    uR = np.asarray(dehom_rm.disp_at_points(B, xy, beam_force_vabs=FF)) * 1e3
    sRg = np.asarray(dehom_rm.stress_at_points(B, sm_xy, beam_force_vabs=FF, frame="material")["stress"]) / 1e6
    np.savez(_cache, uR=uR, sRg=sRg)
    print("computed + cached RM (disp + %d gauss stress)" % len(sm_xy), flush=True)
eg = tri.get_trifinder()(sm_xy[:, 0], sm_xy[:, 1])
ordered = eg.size == 3 * M and np.array_equal(eg.reshape(M, 3), np.arange(M)[:, None].repeat(3, 1))
if ordered:                                                          # .SM = 3 gauss per element, in order
    gxy = sm_xy.reshape(M, 3, 2)
    Ai = np.linalg.inv(np.concatenate([np.ones((M, 3, 1)), gxy], 2))          # inv[1 x y] at gauss pts
    Cc = np.concatenate([np.ones((M, 3, 1)), xy[tris]], 2)                    # [1 x y] at corners
    cornerV = Cc @ (Ai @ sVg.reshape(M, 3, 6)); cornerR = Cc @ (Ai @ sRg.reshape(M, 3, 6))
else:                                                               # robust fallback
    from collections import defaultdict
    gpe = defaultdict(list)
    for gi, e in enumerate(eg):
        if e >= 0 and len(gpe[e]) < 3:
            gpe[e].append(gi)
    cornerV = np.zeros((M, 3, 6)); cornerR = np.zeros((M, 3, 6)); kd = cKDTree(sm_xy)
    for e in range(M):
        cxy = xy[tris[e]]; gl = gpe.get(e, [])
        if len(gl) == 3:
            A = np.column_stack([np.ones(3), sm_xy[gl, 0], sm_xy[gl, 1]]); Cc = np.column_stack([np.ones(3), cxy[:, 0], cxy[:, 1]])
            cornerV[e] = Cc @ np.linalg.solve(A, sVg[gl]); cornerR[e] = Cc @ np.linalg.solve(A, sRg[gl])
        else:
            j = kd.query(cxy.mean(0))[1]; cornerV[e] = sVg[j]; cornerR[e] = sRg[j]
print("gauss->corner extrapolation done (ordered=%s)" % ordered, flush=True)

# clamp each element's extrapolated corners to that element's OWN gauss-point range: a linear fit through
# 3 interior gauss points overshoots on the distorted sliver elements at the web/cap T-junctions, painting
# spurious hot spots there (both VABS and RM share this step) -- clamping removes the junction leakage.
gV6 = sVg.reshape(M, 3, 6); gR6 = sRg.reshape(M, 3, 6)
cornerV = np.clip(cornerV, gV6.min(1, keepdims=True), gV6.max(1, keepdims=True))
cornerR = np.clip(cornerR, gR6.min(1, keepdims=True), gR6.max(1, keepdims=True))
print("clamped corners to per-element gauss range (junction overshoot removed)", flush=True)

# EXPLODED triangulation for the stress contours (each element keeps its own 3 nodes)
EP = xy[tris].reshape(-1, 2)
etri = mtri.Triangulation(EP[:, 0], EP[:, 1], np.arange(3 * M).reshape(M, 3))
sVe = cornerV.reshape(-1, 6); sRe = cornerR.reshape(-1, 6)           # (3M,6) exploded corner stress


def tstress(ax, cornerval, m):                                       # exploded -> discontinuous, no bleed
    cs = ax.tripcolor(etri, np.clip(cornerval, -m, m), shading="gouraud", cmap=CMAP, vmin=-m, vmax=m)
    ax.set_aspect("equal"); ax.axis("off"); return cs


def tdisp(ax, val, m):                                               # nodal -> continuous
    cs = ax.tripcolor(tri, np.clip(val, -m, m), shading="gouraud", cmap=CMAP, vmin=-m, vmax=m)
    ax.set_aspect("equal"); ax.axis("off"); return cs


# ---- stress contours (exploded) : VABS (left) vs RM shell (right), enlarged, headers once ----
STRS = [(0, r"$\sigma_{11}$"), (1, r"$\sigma_{22}$"), (5, r"$\sigma_{12}$")]
fig, ax = plt.subplots(3, 2, figsize=(13.5, 12.0))
for rr, (ci, lab) in enumerate(STRS):
    m = np.nanpercentile(np.abs(np.r_[sVe[:, ci], sRe[:, ci]]), 99) or 1e-9
    for c, dat in enumerate([sVe[:, ci], sRe[:, ci]]):
        cs = tstress(ax[rr, c], dat, m)
    ax[rr, 0].text(-0.05, 0.5, lab, transform=ax[rr, 0].transAxes, rotation=90, va="center", ha="right", fontsize=20)
    cb = fig.colorbar(cs, ax=ax[rr, :].tolist(), shrink=0.85, pad=0.015)
    cb.set_label("MPa", fontsize=20); cb.ax.tick_params(labelsize=18)
ax[0, 0].set_title("VABS", fontsize=22, pad=10); ax[0, 1].set_title("RM shell", fontsize=22, pad=10)
fig.savefig(os.path.join(FIG, "r020_stress_contour_cmp.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote r020_stress_contour_cmp.png", flush=True)

# ---- disp contours (nodal) : VABS (left) vs RM shell (right), enlarged, descriptive row labels ----
UDES = [(0, r"$u_1$  (out-of-plane warping)"), (1, r"$u_2$  (edgewise)"), (2, r"$u_3$  (flapwise)")]
fig, ax = plt.subplots(3, 2, figsize=(13.5, 12.0))
for rr, (ci, lab) in enumerate(UDES):
    m = np.nanpercentile(np.abs(np.r_[uV[:, ci], uR[:, ci]]), 99) or 1e-9
    for c, dat in enumerate([uV[:, ci], uR[:, ci]]):
        cs = tdisp(ax[rr, c], dat, m)
    ax[rr, 0].text(-0.05, 0.5, lab, transform=ax[rr, 0].transAxes, rotation=90, va="center", ha="right", fontsize=15)
    cb = fig.colorbar(cs, ax=ax[rr, :].tolist(), shrink=0.85, pad=0.015)
    cb.set_label("mm", fontsize=20); cb.ax.tick_params(labelsize=18)
ax[0, 0].set_title("VABS", fontsize=22, pad=10); ax[0, 1].set_title("RM shell", fontsize=22, pad=10)
fig.savefig(os.path.join(FIG, "r020_disp_contour_cmp.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote r020_disp_contour_cmp.png", flush=True)

# ---- 3-D deformed shape (warp by (u1,u2,u3); out-of-plane = u1) ----
ext = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
umax = max(np.linalg.norm(uV, axis=1).max(), np.linalg.norm(uR, axis=1).max())
sc = 0.30 * ext / max(umax, 1e-9)
fig = plt.figure(figsize=(14, 6))
for k, (uw, tag) in enumerate([(uV, "VABS"), (uR, "RM shell")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d"); ax.set_proj_type("ortho")
    Xo = uw[:, 0] * sc; Yc = xy[:, 0] + uw[:, 1] * sc; Zf = xy[:, 1] + uw[:, 2] * sc
    ts = ax.plot_trisurf(Yc, Zf, Xo, triangles=tris, cmap=CMAP, edgecolor="none", linewidth=0, antialiased=True)
    ts.set_array(np.linalg.norm(uw, axis=1)[tris].mean(1))
    ax.set_title("%s -- section warped by (u1,u2,u3)" % tag, fontsize=11)
    ax.set_xlabel("y2 (chord)"); ax.set_ylabel("y3 (flap)"); ax.set_zlabel("u1 out-of-plane")
    ax.view_init(elev=20, azim=-60); ax.set_box_aspect((3, 2, 1.3))
    fig.colorbar(ts, ax=ax, shrink=0.55, pad=0.03, label="|warp| (mm)")
fig.savefig(os.path.join(FIG, "r020_deformed3d.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote r020_deformed3d.png (out-of-plane u1: VABS %.3f / RM %.3f mm)"
      % (np.abs(uV[:, 0]).max(), np.abs(uR[:, 0]).max()), flush=True)

# ---- EXPLODED .vtk for ParaView (stress per-element discontinuous, disp continuous on corners) ----
import pyvista as pv
ep3 = np.column_stack([EP[:, 0], EP[:, 1], np.zeros(len(EP))])
faces = np.column_stack([np.full(M, 3), np.arange(3 * M).reshape(M, 3)]).astype(np.int64).ravel()
ex = pv.PolyData(ep3, faces)
uVc = uV[tris].reshape(-1, 3); uRc = uR[tris].reshape(-1, 3)         # nodal disp on the exploded corners
for j, nm in enumerate(["S11", "S22", "S33", "S23", "S13", "S12"]):
    ex.point_data["VABS_" + nm] = sVe[:, j]; ex.point_data["OpenSG_RM_" + nm] = sRe[:, j]
for j, nm in enumerate(["u1", "u2", "u3"]):
    ex.point_data["VABS_" + nm] = uVc[:, j]; ex.point_data["OpenSG_RM_" + nm] = uRc[:, j]
ex.point_data["VABS_umag"] = np.linalg.norm(uVc, axis=1); ex.point_data["OpenSG_RM_umag"] = np.linalg.norm(uRc, axis=1)
ex.point_data["VABS_disp"] = uVc; ex.point_data["OpenSG_RM_disp"] = uRc
ex.save(os.path.join(VTK, "r020_dehom_s10.vtk"))
print("wrote EXPLODED vtk/r020_dehom_s10.vtk (%d corners; stress gauss-based, non-bleeding)" % (3 * M), flush=True)

