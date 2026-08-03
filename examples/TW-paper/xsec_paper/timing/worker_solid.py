"""ONE FEniCS-2D-solid tube homogenization in a fresh process (opensg_env_v8 / dolfinx 0.8.0).

Usage:  python worker_solid.py <case_name> <solid_yaml_path>

Exactly the entry points the reproduce_6x6 solid references were built with
(single_cell_tube/scripts/sweep_solid.py, two_cell_tube/tube2cell_solid*.py):
    sm = SolidBounMesh(yaml);  mp, _ = sm.material_database
    C6 = compute_timo_boun(mp, sm.meshdata)[0]          # opensg FEniCS fork

Prints one machine-readable line:  ###JSON### {...}
"""
import json
import os
import sys
import time

CASE, YAML = sys.argv[1], os.path.abspath(sys.argv[2])

PKG = None
for c in ("~/claude_tmp/opensg-FEniCS", "~/claude_tmp/OpenSG-1.0"):
    c = os.path.expanduser(c)
    if os.path.isfile(os.path.join(c, "opensg", "core", "solid.py")):
        PKG = c
        break
if PKG is None:
    sys.exit("FEniCS opensg fork not found under ~/claude_tmp/")
sys.path.insert(0, PKG)
os.environ.pop("C1_PENALTY", None)

scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scratch", CASE)
os.makedirs(scratch, exist_ok=True)
os.chdir(scratch)                                  # SolidBounMesh drops SG_mesh.msh/.xdmf here

ne = 0                                             # element count = rows under `elements:`
inside = False
for line in open(YAML):
    if line.startswith("elements:"):
        inside = True
        continue
    if inside:
        if line.lstrip().startswith("- "):
            ne += 1
        elif line.strip():
            break

t0 = time.perf_counter()
import numpy as np
import opensg  # noqa: F401
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun
t_import = time.perf_counter() - t0

t1 = time.perf_counter()
sm = SolidBounMesh(YAML)
mp, _ = sm.material_database
C6 = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
t_compute = time.perf_counter() - t1

C6 = 0.5 * (C6 + C6.T)
print("###JSON### " + json.dumps(dict(
    case=CASE, model="solid", n_elem=int(ne), t_import=t_import,
    t_compute=t_compute, t_warm=None,
    diag=[float(C6[i, i]) for i in range(6)])))
