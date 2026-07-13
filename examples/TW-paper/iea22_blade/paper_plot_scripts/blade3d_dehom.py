"""blade3d_dehom.py -- render the IEA-22 blade in 3-D as a loft of the spanwise cross-sections we
computed, coloured by the DEHOMOGENIZED axial stress sigma_11 recovered by the RM shell under each
station's own beam load, interpolated between adjacent windIO airfoil stations.  One colour bar
spans the whole blade.
  -> figures/blade3d_sigma11.png   (and blade3d_umag.png for |u|)
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IO = os.path.join(REPO, "third_party", "OpenSG_io")
sys.path.insert(0, REPO); sys.path.insert(0, IO); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm
from opensg_io import load_blade, build_cross_section

WINDIO = os.path.join(REPO, "examples", "data", "windio", "IEA-22-280-RWT.yaml")
D2 = os.path.join(REPO, "examples", "data", "2d_yaml")
VABS = os.path.join(D2, "IEA_VABS")
IB = os.path.join(REPO, "examples", "TW-paper", "iea22_blade", "data")
Y1 = os.path.join(REPO, "examples", "data", "1d_yaml", "IEA")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
CMAP = plt.cm.rainbow
N = 160          # resampled points around the airfoil
MPER = 14        # interpolated span samples per station interval

STATIONS = [("r020", 0.2000, os.path.join(IB, "shell_r020.yaml")),
            ("r0247", 0.2470, os.path.join(Y1, "shell_r0247.yaml")),
            ("r0399", 0.3993, os.path.join(Y1, "shell_r0399.yaml")),
            ("r0534", 0.5336, os.path.join(Y1, "shell_r0534.yaml")),
            ("r0739", 0.7389, os.path.join(Y1, "shell_r0739.yaml")),
            ("r0980", 0.9800, os.path.join(Y1, "shell_r0980.yaml"))]


def parse_glb(path):
    L = [ln.split() for ln in open(path).read().splitlines() if ln.strip()]
    a = [float(x) for x in L[4]]; b = [float(x) for x in L[5]]
    return np.array([a[0], b[0], b[1], a[1], a[2], a[3]])


def resample_ring(xy, f, n):
    """Resample a closed contour + its scalar field to n points by normalized arc length."""
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]; f = f[:-1]
    c = np.vstack([xy, xy[0]]); fc = np.r_[f, f[0]]
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))]; d /= d[-1]
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.column_stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])]), np.interp(t, d, fc)


blade = load_blade(WINDIO)
rings = []                         # (r, xy(N,2), sig11(N,), umag(N,))
print("station    r      sigma11[min,max] MPa   |u|max mm")
for tag, r, shell in STATIONS:
    B = dehom_rm.build_rm_bundle(shell, ref="oml")
    FF = parse_glb(os.path.join(VABS, "iea_%s.sg.glb" % tag))
    oml = np.asarray(build_cross_section(blade, r=r)["nodes"], float)      # OML contour, ring frame
    sig = np.asarray(dehom_rm.stress_at_points(B, oml, beam_force_vabs=FF, frame="global")["stress"])[:, 0] / 1e6
    umag = np.linalg.norm(np.asarray(dehom_rm.disp_at_points(B, oml, beam_force_vabs=FF)), axis=1) * 1e3
    xy_r, sig_r = resample_ring(oml, sig, N)
    _, um_r = resample_ring(oml, umag, N)
    rings.append((r, xy_r, sig_r, um_r))
    print("  %-6s %.4f   [%7.2f, %7.2f]      %.3f" % (tag, r, sig.min(), sig.max(), umag.max()))


def loft_field(field_idx):
    """Interpolate geometry + the chosen field (2=sigma11, 3=|u|) across the span."""
    span = []
    for k in range(len(rings) - 1):
        r0, xy0, *f0 = rings[k]; r1, xy1, *f1 = rings[k + 1]
        for m in range(MPER):
            t = m / MPER
            span.append((r0 + t * (r1 - r0), (1 - t) * xy0 + t * xy1,
                         (1 - t) * f0[field_idx - 2] + t * f1[field_idx - 2]))
    span.append((rings[-1][0], rings[-1][1], rings[-1][field_idx]))   # rings = (r, xy, sig11, umag)
    return span


def render(field_idx, label, out, symmetric):
    span = loft_field(field_idx)
    allf = np.concatenate([s[2] for s in span])
    if symmetric:
        m = np.nanpercentile(np.abs(allf), 99.5); vmin, vmax = -m, m
    else:
        vmin, vmax = 0.0, np.nanpercentile(allf, 99.5)
    norm = Normalize(vmin, vmax)
    quads, cols = [], []
    for j in range(len(span) - 1):
        xj, xyj, fj = span[j]; xj1, xyj1, fj1 = span[j + 1]
        for i in range(N):
            i1 = (i + 1) % N
            quads.append([(xj, xyj[i, 0], xyj[i, 1]), (xj, xyj[i1, 0], xyj[i1, 1]),
                          (xj1, xyj1[i1, 0], xyj1[i1, 1]), (xj1, xyj1[i, 0], xyj1[i, 1])])
            cols.append(0.25 * (fj[i] + fj[i1] + fj1[i1] + fj1[i]))
    fig = plt.figure(figsize=(15, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    pc = Poly3DCollection(quads, facecolors=CMAP(norm(np.clip(cols, vmin, vmax))),
                          edgecolors="none", linewidths=0, shade=False)
    ax.add_collection3d(pc)
    # station outlines (where the real windIO data lives)
    for r, xy, *_ in rings:
        seg = [[(r, xy[i, 0], xy[i, 1]), (r, xy[(i + 1) % N, 0], xy[(i + 1) % N, 1])] for i in range(N)]
        ax.add_collection3d(Line3DCollection(seg, colors="k", linewidths=0.5, alpha=0.35))
    yall = np.concatenate([rr[1][:, 0] for rr in rings])
    zall = np.concatenate([rr[1][:, 1] for rr in rings])
    ax.set_xlim(0.18, 1.0)
    ax.set_ylim(yall.min() - 0.3, yall.max() + 0.3)
    ax.set_zlim(zall.min() - 0.3, zall.max() + 0.3)
    ax.set_box_aspect((9.0, 5.0, 1.7))
    ax.view_init(elev=20, azim=-72)
    ax.set_xlabel(r"spanwise position $r$", labelpad=12)
    ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_yticks([]); ax.set_zticks([])                 # chord/thickness self-evident; keep only span axis
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0)); ax.zaxis.line.set_color((1, 1, 1, 0))
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.0, aspect=18, location="right")
    cb.set_label(label, fontsize=11)
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out), "range [%.2f, %.2f]" % (vmin, vmax))


render(2, r"$\sigma_{11}$ (MPa)", os.path.join(FIG, "blade3d_sigma11.png"), symmetric=True)
render(3, r"$|u|$ (mm)", os.path.join(FIG, "blade3d_umag.png"), symmetric=False)
