"""time_table9.py -- final cost table (3 cases): square/circle thick m45, webbed
ellipse m45.  Reports shell & solid #dofs and per-stage wall times (optimized).
Shell dofs = 6 * n_shell_nodes ; solid dofs = 3 * n_solid_nodes.
"""
import os, sys, time, io, contextlib, json
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))

import run_indep as ri
from boundary_from_yaml import extract
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness

CLoad = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def shell_nodes(yamlf):
    d = yaml.load(open(yamlf), Loader=CLoad)
    return len(d["nodes"])


def solid_nodes(yamlf):
    d = yaml.load(open(yamlf), Loader=CLoad)
    return len(d["nodes"])


def time_shell(mdir, tag, reps=2):
    res = os.path.join(HERE, "out", "t9_res"); os.makedirs(res, exist_ok=True)
    r = None
    for _ in range(reps):                          # warm the JIT; keep last
        r = ri.shell_solve_lagrange(tag, mdir, res, return_full=True)
    return r


def time_solid(yamlf):
    t0 = time.time(); sm = SolidSegmentMesh(yamlf); t1 = time.time()
    mp, dens = sm.material_database
    compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)
    t2 = time.time()
    compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=True)
    t3 = time.time()
    return t1 - t0, t2 - t1, t3 - t2


rows = []
# square / circle thick m45 : shell 48x10 mesh, solid 48x10x4
for sub, geom in (("taper_square", "square"), ("taper_study", "circle")):
    mdir = os.path.join(HERE, "out", sub, "meshes")
    tag = "thick_m45_aR070"
    syaml = os.path.join(mdir, "shell_%s.yaml" % tag)
    solyaml = os.path.join(mdir, "solid_%s.yaml" % tag)
    sn = shell_nodes(syaml); qn = solid_nodes(solyaml)
    r = time_shell(mdir, tag)
    sl, sb, st = time_solid(solyaml)
    rows.append((geom + " thick m45", 6 * sn, r["t_extract"], r["t_rings"], r["t_seg"],
                 r["t_extract"] + r["t_rings"] + r["t_seg"], 3 * qn, sb, st, sl + sb + st))

# webbed ellipse m45 : shell 48x10 nw6, solid 96x20 nw12 x4
ell = os.path.join(HERE, "out", "ell3w")
syaml = os.path.join(ell, "shell_48x10", "shell_e3w.yaml")
solyaml = os.path.join(ell, "solid_mesh", "solid_e3w.yaml")
sn = shell_nodes(syaml); qn = solid_nodes(solyaml)
r = time_shell(os.path.join(ell, "shell_48x10"), "e3w")
sl, sb, st = time_solid(solyaml)
rows.append(("webbed ellipse m45", 6 * sn, r["t_extract"], r["t_rings"], r["t_seg"],
             r["t_extract"] + r["t_rings"] + r["t_seg"], 3 * qn, sb, st, sl + sb + st))

print("\n%-20s %8s %6s %6s %6s %6s | %8s %6s %6s %6s"
      % ("case", "s.dofs", "extr", "rings", "seg", "total", "q.dofs", "boun", "taper", "total"))
for nm, sd, ex, rg, sg, tt, qd, sb, st, qt in rows:
    print("%-20s %8d %6.1f %6.1f %6.1f %6.1f | %8d %6.1f %6.1f %6.1f"
          % (nm, sd, ex, rg, sg, tt, qd, sb, st, qt))
