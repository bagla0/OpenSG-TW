"""Decisive covariance test (per opensg-msg-expert): rotate the tube mesh about the
beam axis by phi0 and recompute. The section is axisymmetric, so a CORRECT method
must give GA2=GA3 and values invariant to phi0. If KL's GA2/GA3 split changes with
phi0, the asymmetry is a discretization artifact (non-covariant shear condensation),
not a Kirchhoff-model limitation."""
import os
import sys
import numpy as np
import yaml as _yaml

TUBE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_45_45\scripts"
sys.path.insert(0, TUBE)
import tube_lib as T
from gen_meshes import ANI

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\data"
R_MEAN = 0.0715
T_WALL = 0.008682
R_REF = R_MEAN            # centric
D_SHIFT = T_WALL / 2.0
N = 200
LAYUP = [(-45.0, T_WALL)]


class FlowList(list):
    pass


_yaml.add_representer(FlowList, lambda d, data: d.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=True))


def gen_rot(path, R_ref, phi0, layup=LAYUP, mat=ANI, n=N):
    ang = np.array([2.0 * np.pi * k / n + phi0 for k in range(n)])
    pts = [(R_ref * np.cos(t), R_ref * np.sin(t)) for t in ang]
    elems = [(k + 1, k + 2) for k in range(n - 1)] + [(n, 1)]
    ori = []
    for (a, b) in elems:
        p1 = np.array(pts[a - 1]); p2 = np.array(pts[b - 1])
        t = p2 - p1; e2 = t / (np.linalg.norm(t) + 1e-30)
        mid = 0.5 * (p1 + p2); e3 = -mid / (np.linalg.norm(mid) + 1e-30)
        ori.append([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0,
                    float(e3[0]), float(e3[1]), 0.0])
    data = {
        "nodes": [FlowList(["%.10f %.10f 0.0" % (x, y)]) for (x, y) in pts],
        "elements": [FlowList(["%d %d" % (a, b)]) for (a, b) in elems],
        "sets": {"element": [{"name": "tube", "labels": list(range(1, n + 1))}]},
        "sections": [{"type": "shell", "elementSet": "tube",
                      "layup": [["mat", float(t), float(a)] for a, t in layup]}],
        "materials": [{"name": "mat", "density": 1860.0, "elastic": mat}],
        "elementOrientations": [FlowList([float(v) for v in o]) for o in ori],
    }
    with open(path, "w") as f:
        _yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    return path


print("phi0[deg] |  KL: GA2        GA3       (GA2-GA3)/avg  |  RM: GA2        GA3")
for deg in (0.0, 15.0, 30.0, 45.0, 90.0):
    p = os.path.join(DATA, "shell_rot_%03d.yaml" % int(deg))
    gen_rot(p, R_REF, np.deg2rad(deg))
    RM, KF = T.homog(p, R_REF, D_SHIFT, k22_mode="exact")
    RM = 0.5 * (RM + RM.T); KF = 0.5 * (KF + KF.T)
    asym = 100.0 * (KF[1, 1] - KF[2, 2]) / (0.5 * (KF[1, 1] + KF[2, 2]))
    print("%8.0f  | %.4e  %.4e   %+7.2f%%      | %.4e  %.4e"
          % (deg, KF[1, 1], KF[2, 2], asym, RM[1, 1], RM[2, 2]))
    try:
        os.remove(p)
    except OSError:
        pass
