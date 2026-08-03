"""ONE (case, model) tube shell homogenization in a fresh process -- reproduce_6x6 entry points.

Usage:  python worker_shell.py <case_name> <RM|KL>

Uses exactly the entry points the reproduce_6x6 package uses:
  single-cell (exact k22)  RM : tube_lib -> timoshenko_rm(..., p=1, shear="mitc_both")
  single-cell (exact k22)  KL : tube_lib._kirchhoff (Hermite C1)
  two-cell   (geometric)   RM : opensg_jax.fe_jax.strip_RM.rm_timoshenko_6x6(curved=True)
  two-cell   (geometric)   KL : opensg_jax.fe_jax.gradient_kirchhoff.gradient_junction_kirchhoff

Prints one machine-readable line:  ###JSON### {...}
  t_import  : module import time (numpy/jax/opensg_jax/tube_lib)
  t_compute : cold first call in this fresh process (includes JAX JIT compilation)
  t_warm    : identical second call in the same process (JIT cached)
"""
import json
import os
import sys
import time

CASE, MODEL = sys.argv[1], sys.argv[2]
R6 = os.path.expanduser("~/OpenSG-TW-claude/examples/TW-paper/reproduce_6x6")
sys.path.insert(0, R6)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

t0 = time.perf_counter()
import numpy as np
import common                      # bootstraps repo root + reproduce_6x6/lib on sys.path
import jax
jax.config.update("jax_enable_x64", True)
case = [c for c in common.cases() if c["name"] == CASE][0]
if case["method"] == "exact":
    import tube_lib as T
elif MODEL == "RM":
    from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
else:
    from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
t_import = time.perf_counter() - t0


def run_once():
    if case["method"] == "exact":                  # single-cell smooth circle, exact k22 = -1/R
        n3d, elements, mat_db, layup_db, e2l = T.load_yaml(case["mesh"])
        nodes, cells, lpe = T.read_mesh(n3d, elements, e2l)
        nodes2d = nodes[:, :2]
        elems = cells[:, [0, 1]]
        ne = len(elems)
        xy = nodes2d[elems[:, 0]]
        area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                                  - np.roll(xy[:, 0], -1) * xy[:, 1]))
        ksign = -1.0 if area > 0 else 1.0
        k22 = (ksign / case["R"]) * np.ones(ne)

        def D_of(i):
            a = np.asarray(T.compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
            return T.shift_abd_reference(a, case["dshift"]) if case["dshift"] else a

        D_by = {ln: D_of(i) for ln, i in layup_db.items()}
        if MODEL == "RM":
            G_by = {ln: T.transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
                    for ln, i in layup_db.items()}
            M = T.timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1, shear="mitc_both")[0]
        else:
            M = T._kirchhoff(nodes2d, elems, lpe, D_by, k22)
        return np.asarray(M), ne
    if MODEL == "RM":                              # two-cell webbed tube, geometric k22
        M = rm_timoshenko_6x6(case["mesh"], 0.0, dshift=case["dshift"], curved=True, orient=False)
    else:
        M = gradient_junction_kirchhoff(case["mesh"], frac=0.0, dshift=case["dshift"], orient=False)[0]
    return np.asarray(M), None


t1 = time.perf_counter()
M, ne = run_once()
t_compute = time.perf_counter() - t1

t2 = time.perf_counter()
run_once()
t_warm = time.perf_counter() - t2

if ne is None:                                     # element count = rows under `elements:` in the yaml
    ne = 0
    inside = False
    for line in open(case["mesh"]):
        if line.startswith("elements:"):
            inside = True
            continue
        if inside:
            if line.lstrip().startswith("- "):
                ne += 1
            elif line.strip():
                break

M = common.sym(M)
print("###JSON### " + json.dumps(dict(
    case=CASE, model=MODEL, n_elem=int(ne), t_import=t_import,
    t_compute=t_compute, t_warm=t_warm,
    diag=[float(M[i, i]) for i in range(6)])))
