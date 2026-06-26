"""
Benchmark visualizations for the OML 3D-stress comparison.

Produces three PNGs in outputs/:
  oml_stress_6comp.png : 6 separate stress-component plots (S11..S12), MSG-TW
                         (JAX) vs FEniCS solid, vs the non-dimensional OML path
                         parameter s in [0,1] (0 = leading edge -> 1 = trailing
                         edge), the arc-length parameter that sweeps y3.
  oml_path_tw.png      : the MSG-TW (1Dshell) cross-section mesh with the OML
                         stress path marked by arrows (LE -> TE).
  oml_path_solid.png   : the 2Dsolid mesh with the same OML path by arrows.

Run benchmark_oml_jax.py and the WSL FEniCS run first (they write the data).
"""
import os
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
sys.path.insert(0, os.path.dirname(__file__))
from fe_jax import load_yaml, read_mesh
from benchmark_oml_compare import (load, solid_oml_coords, match_outer,
                                   upper_te_to_le, COMP, SOLID)

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
SHELL = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_0.yaml"


def arclen_param(xy):
    """Normalized cumulative arc length in [0,1] along an ordered path."""
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    return d / d[-1]


# ======================================================================= data
jxy, js = load(os.path.join(OUT, "oml_jax.txt"))
oml = solid_oml_coords(SOLID)
idx = upper_te_to_le(jxy)                              # UPPER surface, TE -> LE
jxy, js = jxy[idx], js[idx]
fxy, fs = load(os.path.join(OUT, "oml_fenics_gauss.txt"))
near = np.array([np.min(np.hypot(oml[:, 0] - p[0], oml[:, 1] - p[1])) < 0.05 for p in fxy])
fs_m, _ = match_outer(jxy, fxy[near], fs[near], oml)
s = arclen_param(jxy)                                  # 0 = TE -> 1 = LE


# ============================================== (1) 6-component stress figure
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Upper-surface OML 3D stress (TE->LE)  —  MSG-TW (JAX) vs FEniCS "
             "solid  |  FF=[1e5,5e4,5e4,5e4,1e5,1e5]", fontsize=13, fontweight="bold")
s11scale = np.max(np.abs(js[:, 0]))
for j, c in enumerate(COMP):
    ax = axes.flat[j]
    ax.plot(s, js[:, j] / 1e6, "b-o", ms=4, lw=1.8, label="MSG-TW (JAX)")
    ax.plot(s, fs_m[:, j] / 1e6, "r--s", ms=3, lw=1.5, label="FEniCS solid")
    ax.set_title(f"$\\sigma_{{{c[1:]}}}$  ({c})", fontsize=12, fontweight="bold")
    ax.set_xlabel("upper-surface path  s  (0 = TE  ->  1 = LE)")
    ax.set_ylabel(f"{c}  (MPa)")
    ax.grid(True, ls=":", alpha=0.7)
    ax.legend(fontsize=9)
    mx = max(np.max(np.abs(js[:, j])), np.max(np.abs(fs_m[:, j])))
    note = f"max|.| = {mx / s11scale * 100:.1f}% of S11"
    if c in ("S22", "S33", "S23"):
        note += "\n(free-surface ~0; solid sampled sub-surface)"
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7,
            color="gray", va="bottom")
fig.tight_layout(rect=[0, 0, 1, 0.96])
p6 = os.path.join(OUT, "oml_stress_6comp.png")
fig.savefig(p6, dpi=150); plt.close(fig)
print("wrote", p6)


# ====================================================== pyvista mesh + path
import pyvista as pv
pv.OFF_SCREEN = True


def path_arrows(plotter, opath, every, mag, color):
    pts = np.c_[opath, np.zeros(len(opath))]
    cent = pts[:-1][::every]
    dirs = np.diff(pts, axis=0)[::every]
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)
    plotter.add_arrows(cent, dirs, mag=mag, color=color)
    plotter.add_points(pts[::every], color=color, point_size=7,
                       render_points_as_spheres=True)


def render(poly, opath, title, fname, edge=False):
    p = pv.Plotter(off_screen=True, window_size=(1300, 1000))
    p.set_background("white")
    p.add_mesh(poly, color="lightsteelblue", show_edges=edge,
               edge_color="gray", line_width=2)
    path_arrows(p, opath, every=max(1, len(opath) // 40), mag=0.16, color="red")
    p.add_text(title, font_size=12, color="black")
    p.view_xy()
    p.camera.zoom(1.3)
    p.screenshot(fname)
    print("wrote", fname, os.path.getsize(fname), "bytes")


# --- TW (1Dshell) line mesh; the reference nodes ARE the OML ---
n3, el, mat, lay, e2l = load_yaml(SHELL)
nodes, cells, _ = read_mesh(n3, el, e2l)
tw_pts = np.c_[nodes, np.zeros(len(nodes))]
lines = np.hstack([[2, int(c[0]), int(c[-1])] for c in cells]).astype(np.int64)
tw_poly = pv.PolyData(tw_pts, lines=lines)
render(tw_poly, jxy, "MSG-TW (1Dshell) mesh  |  upper-surface stress path (TE->LE)",
       os.path.join(OUT, "oml_path_tw.png"), edge=True)

# --- 2Dsolid quad mesh ---
def _row(r):
    if isinstance(r, str):
        return r.strip("[]").split()
    if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
        return r[0].strip("[]").split()
    return [str(v) for v in r]
with open(SOLID) as f:
    vd = yaml.safe_load(f)
vN = np.array([[float(v) for v in _row(n)] for n in vd["nodes"]])[:, :2]
velems = [[int(v) for v in _row(e)] for e in vd["elements"]]
sv_pts = np.c_[vN, np.zeros(len(vN))]
faces = np.hstack([[4, e[0] - 1, e[1] - 1, e[2] - 1, e[3] - 1] for e in velems]).astype(np.int64)
sv_poly = pv.PolyData(sv_pts, faces=faces)
render(sv_poly, jxy, "2Dsolid mesh  |  upper-surface stress path (TE->LE)",
       os.path.join(OUT, "oml_path_solid.png"), edge=True)
