"""GA3 mesh-refinement convergence (mh104 f=0.2, CCW, k22=0).  Uniformly subdivide every shell
element into k flat sub-segments (same layup + orientation), homogenize, and watch the diagonal Timo
terms -- especially GA3 -- vs k.  If GA3 plateaus, the residual is the Kirchhoff-shell model limit,
not discretization; if it converges toward the solid, it was faceting/FEM resolution."""
import os
import sys
import numpy as np
import yaml

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "shell_ref_f020_connect.yaml")
S = np.loadtxt(os.path.join(CC, "mh104_thickness_study", "results", "C6_solid_f020.txt"))
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def _row(r):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)]


def _ir(r):
    return [int(float(v)) for v in (str(r[0]).split() if len(r) == 1 else r)]


def refine(k):
    d = yaml.safe_load(open(BASE))
    nodes = [_row(r) for r in d["nodes"]]
    elems = [_ir(r) for r in d["elements"]]
    oris = [_row(r) for r in d["elementOrientations"]]
    eset = {}
    for s in d["sets"]["element"]:
        for lab_ in s["labels"]:
            eset[int(lab_)] = s["name"]
    nodes_o = [list(n) for n in nodes]
    elems_o, oris_o, setlbl = [], [], {}
    for ei, (a, b) in enumerate(elems):
        Pa, Pb = np.array(nodes[a - 1]), np.array(nodes[b - 1])
        chain = [a]
        for j in range(1, k):
            P = Pa + (j / k) * (Pb - Pa)
            nodes_o.append([float(P[0]), float(P[1]), 0.0]); chain.append(len(nodes_o))
        chain.append(b)
        for x, y in zip(chain[:-1], chain[1:]):
            elems_o.append([x, y]); oris_o.append(oris[ei])
            setlbl.setdefault(eset[ei + 1], []).append(len(elems_o))
    out = dict(nodes=nodes_o, elements=elems_o,
               sets={"element": [{"name": nm, "labels": setlbl[nm]} for nm in setlbl]},
               sections=d["sections"], elementOrientations=oris_o, materials=d["materials"])
    p = os.path.join(HERE, "_refine_k%d.yaml" % k)
    yaml.dump(out, open(p, "w"), default_flow_style=None)
    return p, len(elems_o)


print("solid GA3=%.4e  GJ=%.4e  EI2=%.4e" % (S[2, 2], S[3, 3], S[4, 4]))
print(" k   n_elem   GA3 (%diff)    GJ      EI2     GA2     EA")
for k in (1, 2, 4, 8):
    p, ne = refine(k)
    C = np.asarray(solve_tw_from_yaml(p, frac=0.0)["Timo"]); C = 0.5 * (C + C.T)
    dd = [100 * (C[i, i] - S[i, i]) / abs(S[i, i]) for i in range(6)]
    print(" %2d  %5d   %+6.1f%%      %+5.1f%%  %+5.1f%%  %+5.1f%%  %+5.1f%%" % (
        k, ne, dd[2], dd[3], dd[4], dd[1], dd[0]))
