"""time_boundary_3cases.py -- BOUNDARY (cross-section) homogenization for the 3
thick m45 cases: RM shell ring vs FEniCS 3-D solid boundary.

Both compute the two end cross-sections' Timoshenko 6x6:
  shell : 2x ring_indep           (6*m warping DOF per ring)
  solid : compute_stiffness(Taper=False) = compute_timo_boun on l+r submeshes
          (3*nodes per boundary submesh; nr=4 through-thickness)
Warm-repeat timed to drop one-time JIT.  For the webbed ellipse, additionally
dump the full L-ring 6x6 (shell vs solid) for the boundary-stiffness table.
"""
import os, sys, io, json, time, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))

from boundary_from_yaml import extract
from segment_element import compute_k22
from solve_segment_jax import _material_by_section
from run_ring_indep import ring_indep
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness

OUT = os.path.join(HERE, "out", "b3"); os.makedirs(OUT, exist_ok=True)


def shell_boundary(shell_yaml, reps=2, want_C=False):
    npz = os.path.join(OUT, os.path.basename(shell_yaml).replace(".yaml", ".npz"))
    with contextlib.redirect_stdout(io.StringIO()):
        extract(shell_yaml, npz, plot=False)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"]); cross = [j for j in range(3) if j != ax]
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                      json.loads(str(b["materials"])), center_ref=True)
    A = {}
    for side in ("L", "R"):
        rx = np.asarray(b["%s_x" % side]); rc = np.asarray(b["%s_cells" % side])
        rs = np.asarray(b["%s_subdom" % side]); re3 = np.asarray(b["%s_e3" % side])
        kr = compute_k22(rx[rc].mean(1), np.asarray(b["%s_e2" % side]), re3, rc)
        A[side] = (rx, rc, rs, re3, D_by, G_by, kr, ax, cross)
    m = len(np.asarray(b["L_x"]))
    t = CL = None
    for _ in range(reps):
        t0 = time.perf_counter()
        for side in ("L", "R"):
            C = ring_indep(*A[side])
            if side == "L":
                CL = 0.5 * (np.asarray(C) + np.asarray(C).T)
        t = time.perf_counter() - t0
    return t, 6 * m, (CL if want_C else None)


def solid_boundary(solid_yaml, reps=2):
    sm = SolidSegmentMesh(solid_yaml)
    mp, dens = sm.material_database
    t = res = None
    for _ in range(reps):
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            res = compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)
        t = time.time() - t0
    Ldof = 3 * int(sm.left_submesh["mesh"].geometry.x.shape[0])
    DL = 0.5 * (np.asarray(res[0][0]) + np.asarray(res[0][0]).T)     # L boundary 6x6
    return t, Ldof, DL


rows = []
MESHD = {"square": "taper_square", "circle": "taper_study"}
for geom, sub in MESHD.items():
    mdir = os.path.join(HERE, "out", sub, "meshes"); tg = "thick_m45_aR070"
    ts_t, sdof, _ = shell_boundary(os.path.join(mdir, "shell_%s.yaml" % tg))
    tq_t, qdof, _ = solid_boundary(os.path.join(mdir, "solid_%s.yaml" % tg))
    rows.append(("%s thick m45" % geom, sdof, ts_t, qdof, tq_t))

# ---- thick webbed ellipse (t=0.2) ----
import run_ell3w as e3w
e3w.T = 0.2
edir = os.path.join(OUT, "ell_thick"); os.makedirs(edir, exist_ok=True)
shd = os.path.join(edir, "shell"); sod = os.path.join(edir, "solid")
e3w.gen_ell3w(shd, nc=48, nl=2, nw=6, nr=4)
e3w.gen_ell3w(sod, nc=96, nl=2, nw=12, nr=4)
ts_t, sdof, CL_shell = shell_boundary(os.path.join(shd, "shell_e3w.yaml"), want_C=True)
tq_t, qdof, DL_solid = solid_boundary(os.path.join(sod, "solid_e3w.yaml"))
rows.append(("webbed ellipse thick m45", sdof, ts_t, qdof, tq_t))

print("\n=== BOUNDARY TIME (2 end cross-sections), 3 thick m45 cases ===")
print("%-26s %9s %9s | %9s %9s | %6s"
      % ("case", "shellDOF", "shell s", "solidDOF", "solid s", "x"))
for nm, sd, st, qd, qt in rows:
    print("%-26s %9d %9.2f | %9d %9.2f | %6.1f" % (nm, sd, st, qd, qt, qt / st))

print("\n=== THICK WEBBED ELLIPSE, L-boundary Timoshenko 6x6 (shell vs solid) ===")
LBL = {0: "C11 EA", 1: "C22 GA2", 2: "C33 GA3", 3: "C44 GJ", 4: "C55 EI2", 5: "C66 EI3"}
thr = 1e-3 * np.abs(np.diag(DL_solid)).max()
print("%-10s %14s %14s %9s" % ("term", "solid", "shell", "%err"))
for i in range(6):
    for j in range(i, 6):
        so = DL_solid[i, j]; sh = CL_shell[i, j]
        if abs(so) > thr or abs(sh) > thr:
            nm = LBL[i] if i == j else "C%d%d" % (i + 1, j + 1)
            e = 100 * (sh - so) / so if so != 0 else float("nan")
            print("%-10s %14.5e %14.5e %+8.1f%%" % (nm, so, sh, e))
np.savez(os.path.join(OUT, "ell_thick_boun.npz"), shell=CL_shell, solid=DL_solid)
