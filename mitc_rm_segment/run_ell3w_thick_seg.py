"""run_ell3w_thick_seg.py -- THICK (t=0.2) webbed ellipse SEGMENT (taper): 6-DOF RM
shell (full integration) vs 3-D solid, all nonzero Timoshenko 6x6.  Companion to the
thick-boundary table; same geometry/webs as the thin ell3w segment case."""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))

import run_ell3w as e3w
e3w.T = 0.2                                              # THICK wall
OUT = os.path.join(HERE, "out", "ell3w_thick"); os.makedirs(OUT, exist_ok=True)

from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
import run_indep as ri

# solid reference: 96x20 skin, nw12, x4 through-thickness (same as thin ell3w)
sod = os.path.join(OUT, "solid"); e3w.gen_ell3w(sod, nc=96, nl=20, nw=12, nr=4)
sm = SolidSegmentMesh(os.path.join(sod, "solid_e3w.yaml"))
mp, dens = sm.material_database
So = np.asarray(compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=True)[0])
So = 0.5 * (So + So.T)

# shell: 48x10 skin, six elements per web, full integration
shd = os.path.join(OUT, "shell"); e3w.gen_ell3w(shd, nc=48, nl=10, nw=6, nr=4)
Sh = np.asarray(ri.shell_solve_lagrange("e3w", shd, os.path.join(OUT, "res")))

LBL = {0: "C11 (EA)", 1: "C22 (GA2)", 2: "C33 (GA3)", 3: "C44 (GJ)", 4: "C55 (EI2)", 5: "C66 (EI3)"}
thr = 1e-3 * np.abs(np.diag(So)).max()
print("\n=== THICK (t=0.2) WEBBED ELLIPSE SEGMENT: shell (full int.) vs 3-D solid ===")
print("%-10s %14s %14s %9s" % ("term", "solid", "shell", "%err"))
for i in range(6):
    for j in range(i, 6):
        so = So[i, j]; sh = Sh[i, j]
        if abs(so) > thr or abs(sh) > thr:
            nm = LBL[i] if i == j else "C%d%d" % (i + 1, j + 1)
            e = 100 * (sh - so) / so if so != 0 else float("nan")
            print("%-10s %14.5e %14.5e %+8.1f%%" % (nm, so, sh, e))
print("\nsolid diag x1e9:", np.round(np.diag(So) / 1e9, 5))
print("shell diag x1e9:", np.round(np.diag(Sh) / 1e9, 5))
np.savez(os.path.join(OUT, "seg.npz"), solid=So, shell=Sh)
