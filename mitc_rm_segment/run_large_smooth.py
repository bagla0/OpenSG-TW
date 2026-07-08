"""run_large_smooth.py -- LARGE-SCALE SMOOTH tapered ellipse (no webs, so the shell
refines cleanly -- unlike a folded/webbed section, whose shears drift under
refinement).  ~1.2M-DOF 3-D solid (FEniCS time) vs a refined ~20k-node RM shell
(sparse driver, ~0.12M DOF).  Same [-45] ply, a:1.0->0.65, b:0.6->0.42, L=2.0.
"""
import os, sys, time, math
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))

import run_ellipse as re          # gen_ellipse(mesh_dir, nc, nl, nr) writes shell+solid
CLoad = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
OUT = os.path.join(HERE, "out", "large_smooth"); os.makedirs(OUT, exist_ok=True)

# solid ~1.2M DOF, shell ~20k nodes (converged; smooth wall -> clean refinement)
NC_S, NL_S, NR = 384, 207, 4
NC_H, NL_H = 192, 103


def mem_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 / 1024
    return -1


def gen_shell_only(mesh_dir, nc, nl):
    """Fast shell-only ellipse mesh (avoids gen_ellipse's slow solid dump)."""
    os.makedirs(mesh_dir, exist_ok=True)
    Z = [re.L * i / nl for i in range(nl + 1)]
    snodes, squads, soris = [], [], []
    for z in Z:
        a, b = re.ab(z)
        for k in range(nc):
            th = 2 * math.pi * k / nc
            snodes.append([a * math.cos(th), b * math.sin(th), float(z)])
    for i in range(nl):
        for k in range(nc):
            k1 = (k + 1) % nc
            squads.append([i * nc + k + 1, i * nc + k1 + 1,
                           (i + 1) * nc + k1 + 1, (i + 1) * nc + k + 1])
            a1, a2, e3, _ = re.frame(2 * math.pi * (k + 0.5) / nc, (Z[i] + Z[i + 1]) / 2)
            soris.append(a1.tolist() + a2.tolist() + e3.tolist())
    shell = {"nodes": snodes, "elements": squads,
             "sections": [{"elementSet": "wall", "layup": [["ani", re.T, -45.0]]}],
             "sets": {"element": [{"name": "wall", "labels": list(range(1, len(squads) + 1))}]},
             "materials": [{"name": re.ANI["name"], "density": re.ANI["rho"],
                            "elastic": {"E": re.ANI["E"], "G": re.ANI["G"], "nu": re.ANI["nu"]}}],
             "elementOrientations": soris}
    yaml.safe_dump(shell, open(os.path.join(mesh_dir, "shell_ell.yaml"), "w"),
                   default_flow_style=None, sort_keys=False)
    return len(snodes)


# ---- refined shell (~20k nodes) ----
shd = os.path.join(OUT, "shell")
nsh = gen_shell_only(shd, NC_H, NL_H)
print("shell mesh: %d nodes (%d DOF)" % (nsh, 6 * nsh)); sys.stdout.flush()
import run_indep as ri
t0 = time.time()
rs = ri.shell_solve_lagrange_sparse("ell", shd, os.path.join(OUT, "res"), return_full=True)
tshell = time.time() - t0
Sh = np.asarray(rs["S6"])
print("SHELL (sparse) %d DOF: total %.1fs (extract %.1f rings %.1f seg %.1f)  MemAvail %.1f GB"
      % (rs["ndof"], tshell, rs["t_extract"], rs["t_rings"], rs["t_seg"], mem_gb()))
print("  shell diag (x1e9):", np.round(np.diag(Sh) / 1e9, 5)); sys.stdout.flush()
np.savez(os.path.join(OUT, "shell_result.npz"), S6=Sh, ndof=rs["ndof"], t=tshell,
         te=rs["t_extract"], tr=rs["t_rings"], ts=rs["t_seg"], nnodes=nsh)

# ---- ~1.2M-DOF solid ----
print("\ngen solid nc=%d nl=%d nr=%d (slow yaml dump) ..." % (NC_S, NL_S, NR)); sys.stdout.flush()
sod = os.path.join(OUT, "solid")
t0 = time.time()
re.gen_ellipse(sod, nc=NC_S, nl=NL_S, nr=NR)
print("  solid mesh gen %.0fs, MemAvail %.1f GB" % (time.time() - t0, mem_gb())); sys.stdout.flush()

from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
t0 = time.time()
sm = SolidSegmentMesh(os.path.join(sod, "solid_ell.yaml"))
tload = time.time() - t0
ndof_s = 3 * int(sm.meshdata["mesh"].geometry.x.shape[0])
print("solid: %d DOF, load %.0fs, MemAvail %.1f GB" % (ndof_s, tload, mem_gb())); sys.stdout.flush()
mp, dens = sm.material_database
t0 = time.time()
compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)
tboun = time.time() - t0
print("  solid boundary %.0fs, MemAvail %.1f GB" % (tboun, mem_gb())); sys.stdout.flush()
t0 = time.time()
So = np.asarray(compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=True)[0])
ttap = time.time() - t0
So = 0.5 * (So + So.T)
print("  solid taper %.0fs, MemAvail %.1f GB" % (ttap, mem_gb())); sys.stdout.flush()
np.savez(os.path.join(OUT, "solid_result.npz"), S6=So, ndof=ndof_s,
         tload=tload, tboun=tboun, ttap=ttap)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
print("\n=== LARGE SMOOTH ELLIPSE: shell(%d DOF, %d nodes) vs solid(%d DOF) ==="
      % (rs["ndof"], nsh, ndof_s))
print("%-6s %14s %14s %8s" % ("term", "solid x1e9", "shell x1e9", "%err"))
for i in range(6):
    print("%-6s %14.5f %14.5f %+7.1f%%"
          % (LBL[i], So[i, i] / 1e9, Sh[i, i] / 1e9, 100 * (Sh[i, i] - So[i, i]) / So[i, i]))
print("\nTIMES  shell %.1fs (%d DOF) | solid boun %.0f + taper %.0f = %.0fs (load %.0f, %d DOF)"
      % (tshell, rs["ndof"], tboun, ttap, tboun + ttap, tload, ndof_s))
