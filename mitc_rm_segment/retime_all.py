"""retime_all.py -- WARM OpenSG homogenization-compute times (drop first run for
JIT; exclude runtime mesh construction/extraction).  Reports, per case:
  shell : rings + segment          (the two SG solves; extract excluded)
  solid : boundary + taper         (compute_stiffness; mesh load excluded)
for the tab:cost trio and the large smooth-ellipse case.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))

import run_indep as ri
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness


def shell_warm(tg, mdir, sparse=False):
    fn = ri.shell_solve_lagrange_sparse if sparse else ri.shell_solve_lagrange
    res = os.path.join(HERE, "out", "retime_res"); os.makedirs(res, exist_ok=True)
    fn(tg, mdir, res, return_full=True)                 # cold (warm the JIT)
    r = fn(tg, mdir, res, return_full=True)             # warm
    return r["t_rings"], r["t_seg"]


def solid_warm(solid_yaml):
    sm = SolidSegmentMesh(solid_yaml)                   # mesh construction (excluded)
    mp, dens = sm.material_database
    a = (mp, sm.meshdata, sm.left_submesh, sm.right_submesh)
    compute_stiffness(*a, Taper=False); compute_stiffness(*a, Taper=True)   # warm JIT
    t0 = time.time(); compute_stiffness(*a, Taper=False); tb = time.time() - t0
    t0 = time.time(); compute_stiffness(*a, Taper=True);  tt = time.time() - t0
    return tb, tt


CASES = [
    ("square thick m45", os.path.join(HERE, "out", "taper_square", "meshes"), "thick_m45_aR070",
     os.path.join(HERE, "out", "taper_square", "meshes", "solid_thick_m45_aR070.yaml"), False),
    ("circle thick m45", os.path.join(HERE, "out", "taper_study", "meshes"), "thick_m45_aR070",
     os.path.join(HERE, "out", "taper_study", "meshes", "solid_thick_m45_aR070.yaml"), False),
    ("webbed ellipse m45", os.path.join(HERE, "out", "ell3w", "shell_48x10"), "e3w",
     os.path.join(HERE, "out", "ell3w", "solid_mesh", "solid_e3w.yaml"), False),
    ("large smooth ellipse", os.path.join(HERE, "out", "large_smooth", "shell"), "ell",
     os.path.join(HERE, "out", "large_smooth", "solid", "solid_ell.yaml"), True),
]

print("%-22s | %6s %7s %7s | %7s %7s %7s"
      % ("case (WARM compute)", "s.ring", "s.seg", "s.tot", "q.boun", "q.tap", "q.tot"))
for nm, mdir, tg, syaml, sparse in CASES:
    sr, ss = shell_warm(tg, mdir, sparse=sparse)
    qb, qt = solid_warm(syaml)
    print("%-22s | %6.2f %7.2f %7.2f | %7.2f %7.2f %7.2f"
          % (nm, sr, ss, sr + ss, qb, qt, qb + qt))
    sys.stdout.flush()
