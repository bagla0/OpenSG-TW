"""time_boundary_refine.py -- REFINED boundary (cross-section) homogenization time:
6-DOF RM shell ring vs FEniCS 3-D solid boundary, swept over hoop resolution.

Fixed physical cross-section (prismatic circle R=1, thin iso wall), refined in the
hoop direction nc = 48..768.  Both solvers compute the two end cross-sections'
Timoshenko 6x6:
  shell : 2x ring_indep  (6*nc warping DOF each)
  solid : compute_stiffness(Taper=False) = compute_timo_boun on l+r submeshes
          (3*nc*(nr+1) DOF each; nr=4 through-thickness)
Pure boundary-solve wall time (mesh I/O excluded on both sides); a warm repeat is
timed to drop one-time JIT.
"""
import os, sys, io, json, time, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))

import taper_study as ts
from boundary_from_yaml import extract
from segment_element import compute_k22
from solve_segment_jax import _material_by_section
from run_ring_indep import ring_indep
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness

NR = 4
OUT = os.path.join(HERE, "out", "bref"); os.makedirs(OUT, exist_ok=True)


def shell_rings(mdir, tg, reps=2):
    npz = os.path.join(OUT, "%s.npz" % tg)
    with contextlib.redirect_stdout(io.StringIO()):
        extract(os.path.join(mdir, "shell_%s.yaml" % tg), npz, plot=False)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"]); cross = [j for j in range(3) if j != ax]
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                      json.loads(str(b["materials"])), center_ref=True)
    args = {}
    for side in ("L", "R"):
        rx = np.asarray(b["%s_x" % side]); rc = np.asarray(b["%s_cells" % side])
        rs = np.asarray(b["%s_subdom" % side]); re3 = np.asarray(b["%s_e3" % side])
        kr = compute_k22(rx[rc].mean(1), np.asarray(b["%s_e2" % side]), re3, rc)
        args[side] = (rx, rc, rs, re3, D_by, G_by, kr, ax, cross)
    m = len(np.asarray(b["L_x"]))
    t = None
    for _ in range(reps):
        t0 = time.perf_counter()
        for side in ("L", "R"):
            ring_indep(*args[side])
        t = time.perf_counter() - t0
    return t, 6 * m


def solid_boundary(mdir, tg, reps=2):
    sm = SolidSegmentMesh(os.path.join(mdir, "solid_%s.yaml" % tg))
    mp, dens = sm.material_database
    t = None
    for _ in range(reps):
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)
        t = time.time() - t0
    ndof = 3 * int(sm.left_submesh["mesh"].geometry.x.shape[0])
    return t, ndof


print("REFINED BOUNDARY (cross-section) homogenization: 2 end rings, prismatic circle")
print("%6s | %10s %10s | %10s %10s | %7s"
      % ("nc", "shell DOF", "shell s", "solid DOF", "solid s", "ratio"))
for nc in (48, 96, 192, 384, 768):
    mdir = os.path.join(OUT, "nc%d" % nc)
    ts.gen_case("thin", "iso", 1.0, mesh_dir=mdir, nc=nc, nl=2, nr=NR)
    tg = ts.tag_of("thin", "iso", 1.0)
    ts_time, sdof = shell_rings(mdir, tg)
    tq_time, qdof = solid_boundary(mdir, tg)
    print("%6d | %10d %10.2f | %10d %10.2f | %7.1f"
          % (nc, sdof, ts_time, qdof, tq_time, tq_time / ts_time))
    sys.stdout.flush()
