"""run_large_ellipse.py -- LARGE-SCALE webbed ellipse: ~1.2M-DOF 3-D solid
(FEniCS/MUMPS time) + the corresponding refined RM shell (sparse, ~0.49M DOF).

One gen_ell3w(nc,nl,nw,nr) writes BOTH the refined shell mid-surface mesh and the
conforming solid.  Shell solved with the sparse driver (shell_solve_lagrange_sparse);
solid with the optimized FEniCS solver.  Reports #DOF, wall time, and the diagonal
6x6 (shell vs the 1.2M-DOF solid).
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))

NC, NL, NW, NR = 352, 220, 6, 4
OUT = os.path.join(HERE, "out", "large_ell"); os.makedirs(OUT, exist_ok=True)


def mem_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 / 1024
    return -1


import run_ell3w as e3w
print("gen refined webbed ellipse nc=%d nl=%d nw=%d nr=%d ..." % (NC, NL, NW, NR)); sys.stdout.flush()
t0 = time.time()
e3w.gen_ell3w(OUT, nc=NC, nl=NL, nw=NW, nr=NR)
print("  mesh gen %.0fs, MemAvail %.1f GB" % (time.time() - t0, mem_gb())); sys.stdout.flush()

# ---- refined RM shell (sparse) ----
import run_indep as ri
t0 = time.time()
rs = ri.shell_solve_lagrange_sparse("e3w", OUT, os.path.join(OUT, "res"), return_full=True)
tshell = time.time() - t0
Sh = np.asarray(rs["S6"])
print("\nSHELL (sparse): %d DOF  total %.1fs (extract %.1f rings %.1f seg %.1f)  MemAvail %.1f GB"
      % (rs["ndof"], tshell, rs["t_extract"], rs["t_rings"], rs["t_seg"], mem_gb()))
print("  shell diag (x1e9):", np.round(np.diag(Sh) / 1e9, 5)); sys.stdout.flush()
np.savez(os.path.join(OUT, "shell_result.npz"), S6=Sh, ndof=rs["ndof"],
         t=tshell, te=rs["t_extract"], tr=rs["t_rings"], ts=rs["t_seg"])

# ---- ~1.2M-DOF 3-D solid (FEniCS) ----
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
print("\nloading solid mesh (MemAvail %.1f GB) ..." % mem_gb()); sys.stdout.flush()
t0 = time.time()
sm = SolidSegmentMesh(os.path.join(OUT, "solid_e3w.yaml"))
tload = time.time() - t0
ndof_s = 3 * int(sm.meshdata["mesh"].geometry.x.shape[0])
print("  solid load %.0fs, %d DOF, MemAvail %.1f GB" % (tload, ndof_s, mem_gb())); sys.stdout.flush()
mp, dens = sm.material_database
t0 = time.time()
bnd = compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)[0]
tboun = time.time() - t0
print("  solid boundary %.0fs, MemAvail %.1f GB" % (tboun, mem_gb())); sys.stdout.flush()
t0 = time.time()
So = np.asarray(compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=True)[0])
ttap = time.time() - t0
print("  solid taper %.0fs, MemAvail %.1f GB" % (ttap, mem_gb())); sys.stdout.flush()
So = 0.5 * (So + So.T)
np.savez(os.path.join(OUT, "solid_result.npz"), S6=So, ndof=ndof_s,
         tload=tload, tboun=tboun, ttap=ttap)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
print("\n=== LARGE WEBBED ELLIPSE: shell(%d DOF) vs solid(%d DOF) ===" % (rs["ndof"], ndof_s))
print("%-6s %14s %14s %8s" % ("term", "solid x1e9", "shell x1e9", "%err"))
for i in range(6):
    print("%-6s %14.5f %14.5f %+7.1f%%"
          % (LBL[i], So[i, i] / 1e9, Sh[i, i] / 1e9, 100 * (Sh[i, i] - So[i, i]) / So[i, i]))
print("\nTIMES  shell %.1fs (%d DOF) | solid load %.0f + boun %.0f + taper %.0f = %.0fs (%d DOF)"
      % (tshell, rs["ndof"], tload, tboun, ttap, tload + tboun + ttap, ndof_s))
