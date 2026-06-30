"""Shell-mesh refinement (dmesh) convergence of the JAX-Kirchhoff transverse shear.
Worst-case alignment phi0=0 (nodes on the x2/x3 axes). Exact hoop curvature k22=-1/R.
A correct method -> GA2=GA3 and -> solid as N grows; show the node-on-axis artifact
vanish under refinement. Solid ref GA2=GA3=1.4661e7, EI2=EI3=1.2253e5."""
import os
import sys
import numpy as np
import yaml as _yaml

TUBE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_45_45\scripts"
sys.path.insert(0, TUBE)
import tube_lib as T
from gen_meshes import ANI

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\data"
R_REF = 0.0715
D_SHIFT = 0.008682 / 2.0
LAYUP = [(-45.0, 0.008682)]
S = 0.5 * (np.loadtxt(os.path.join(DATA, "C6_solid_314.txt")) +
           np.loadtxt(os.path.join(DATA, "C6_solid_314.txt")).T)
GA_S, EI_S = S[1, 1], S[4, 4]


class FlowList(list):
    pass


_yaml.add_representer(FlowList, lambda d, data: d.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=True))


def gen(path, n, phi0=0.0):
    ang = np.array([2.0 * np.pi * k / n + phi0 for k in range(n)])
    pts = [(R_REF * np.cos(t), R_REF * np.sin(t)) for t in ang]
    elems = [(k + 1, k + 2) for k in range(n - 1)] + [(n, 1)]
    ori = []
    for (a, b) in elems:
        p1 = np.array(pts[a - 1]); p2 = np.array(pts[b - 1])
        e2 = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-30)
        mid = 0.5 * (p1 + p2); e3 = -mid / (np.linalg.norm(mid) + 1e-30)
        ori.append([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0, float(e3[0]), float(e3[1]), 0.0])
    data = {"nodes": [FlowList(["%.10f %.10f 0.0" % (x, y)]) for (x, y) in pts],
            "elements": [FlowList(["%d %d" % (a, b)]) for (a, b) in elems],
            "sets": {"element": [{"name": "tube", "labels": list(range(1, n + 1))}]},
            "sections": [{"type": "shell", "elementSet": "tube",
                          "layup": [["mat", float(t), float(a)] for a, t in LAYUP]}],
            "materials": [{"name": "mat", "density": 1860.0, "elastic": ANI}],
            "elementOrientations": [FlowList([float(v) for v in o]) for o in ori]}
    with open(path, "w") as f:
        _yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    return path


print("solid GA=%.4e  EI=%.4e  (reference)" % (GA_S, EI_S))
print(" N   | KL GA2      GA3      asym%%   GA%%err |  KL EI2      EI3      asym%%  | RM GA(sym)  GA%%err")
for n in (200, 400, 800, 1600, 3200):
    p = os.path.join(DATA, "_conv.yaml")
    gen(p, n)
    RM, KF = T.homog(p, R_REF, D_SHIFT, k22_mode="exact")
    RM = 0.5 * (RM + RM.T); KF = 0.5 * (KF + KF.T)
    ga_as = 100 * (KF[1, 1] - KF[2, 2]) / (0.5 * (KF[1, 1] + KF[2, 2]))
    ei_as = 100 * (KF[4, 4] - KF[5, 5]) / (0.5 * (KF[4, 4] + KF[5, 5]))
    ga_er = 100 * (0.5 * (KF[1, 1] + KF[2, 2]) - GA_S) / GA_S
    rm_er = 100 * (RM[1, 1] - GA_S) / GA_S
    print("%4d | %.4e %.4e %+5.2f  %+6.2f  | %.4e %.4e %+5.2f  | %.4e %+6.2f"
          % (n, KF[1, 1], KF[2, 2], ga_as, ga_er, KF[4, 4], KF[5, 5], ei_as, RM[1, 1], rm_er))
try:
    os.remove(os.path.join(DATA, "_conv.yaml"))
except OSError:
    pass
