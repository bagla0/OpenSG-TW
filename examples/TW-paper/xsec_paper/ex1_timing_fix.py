"""ex1_timing_fix.py -- patch the two_cell row of timing4.npz.
The stored two-cell 2-D solid mesh mixes triangles at the web junction (breaking the
quad-extrusion FEniCS route) and is unusually heavy (18k dof).  Generate a clean
pure-quad two-cell solid (identical construction to the validated ell4cell generator:
watertight web junction, NR through-thickness layers), VERIFY its 6x6 against the stored
solid reference C6_solid_tube2cell_thin, then time OpenSG-JAX and OpenSG-FEniCS on it.
"""
import math
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
for q in (HERE, os.path.abspath(os.path.join(HERE, "..", "..", "..")),
          os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))):
    sys.path.insert(0, q)
from xsec_5v6_master import load_solid
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
from timing4 import t_fenics_2d, t_jax_yaml

R, T, NR = 0.05, 0.004, 4
NC, NW = 200, 40
E0, NU = 68.9e9, 0.33
MAT = {"name": "iso", "E": [E0] * 3, "G": [E0 / (2 * (1 + NU))] * 3, "nu": [NU] * 3, "rho": 2700.0}
TC = os.path.abspath(os.path.join(HERE, "..", "two_cell_tube", "data"))
OUT = os.path.join(HERE, "results"); SCR = os.path.join(OUT, "_timing4_scr")
os.makedirs(SCR, exist_ok=True)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


class FL(list):
    pass


yaml.add_representer(FL, lambda d, x: d.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))

# ---- pure-quad two-cell solid: annulus + vertical web, watertight at (0,+-R) ----
kt, kb = NC // 4, 3 * NC // 4          # skin node indices at (0,+R) and (0,-R)


def sid(m, k):
    return m * NC + (k % NC)


nodes = []
for m in range(NR + 1):
    off = (m / NR - 0.5) * T
    for k in range(NC):
        th = 2 * math.pi * k / NC
        nodes.append([(R + off) * math.cos(th), (R + off) * math.sin(th), 0.0])
wbase = len(nodes)
for j in range(1, NW):                  # web interior nodes, NR+1 through-thickness (x)
    eta = -1.0 + 2.0 * j / NW
    for m in range(NR + 1):
        off = (m / NR - 0.5) * T
        nodes.append([off, eta * R, 0.0])


def wsid(j, m):
    if j == 0:
        return sid(m, kb)
    if j == NW:
        return sid(m, kt)
    return wbase + (j - 1) * (NR + 1) + m


quads, oris = [], []
for m in range(NR):                     # skin quads (fiber 0 deg: e1=beam, e2=hoop, e3=radial-in)
    for k in range(NC):
        th = 2 * math.pi * (k + 0.5) / NC
        quads.append([sid(m, k) + 1, sid(m, (k + 1)) + 1, sid(m + 1, (k + 1)) + 1, sid(m + 1, k) + 1])
        e2 = [-math.sin(th), math.cos(th), 0.0]
        e3 = [-math.cos(th), -math.sin(th), 0.0]
        oris.append([0.0, 0.0, 1.0] + e2 + e3)
wo = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]      # web: e2=+y, e3=+x
for j in range(NW):
    for m in range(NR):
        quads.append([wsid(j, m) + 1, wsid(j + 1, m) + 1, wsid(j + 1, m + 1) + 1, wsid(j, m + 1) + 1])
        oris.append(list(wo))

solid = {
    "nodes": [FL(["%.10f %.10f %.10f" % tuple(p)]) for p in nodes],
    "elements": [FL([" ".join(str(x) for x in q)]) for q in quads],
    "sets": {"element": [{"name": "iso", "labels": FL(list(range(1, len(quads) + 1)))}]},
    "elementOrientations": [FL(o) for o in oris],
    "materials": [{"name": "iso", "E": FL(MAT["E"]), "G": FL(MAT["G"]),
                   "nu": FL(MAT["nu"]), "rho": MAT["rho"]}],
}
yp = os.path.join(SCR, "solid_twocell_quad.yaml")
yaml.dump(solid, open(yp, "w"), default_flow_style=None, sort_keys=False)
print("generated pure-quad two-cell solid: %d nodes, %d quads" % (len(nodes), len(quads)))

# ---- verify vs the stored solid reference ----
So = load_solid(os.path.join(TC, "C6_solid_tube2cell_thin.txt"))
C = np.asarray(compute_timo_from_yaml(yp, verbose=False)); C = 0.5 * (C + C.T)
err = [100.0 * (C[i, i] - So[i, i]) / So[i, i] for i in range(6)]
print("quad-mesh 6x6 vs stored solid ref, diag %err:",
      "  ".join("%s%+6.2f" % (LBL[i], err[i]) for i in range(6)))
assert max(abs(e) for e in err) < 3.0, "quad solid mesh does not reproduce the reference"

# ---- time both solid solvers on it (FEniCS EA must reproduce the reference) ----
tj1, tj2, dj, _ = t_jax_yaml(yp)
tf, df, ea_fe = t_fenics_2d(yp, "two_cell_quad")
e_fe = 100.0 * (ea_fe - So[0, 0]) / So[0, 0]
print("two_cell(quad): JAX %5.2f/%5.2fs (%d dof) | FEniCS %5.2fs (%d dof) EA err %+.2f%%"
      % (tj1, tj2, dj, tf, df, e_fe))
assert abs(e_fe) < 3.0, "FEniCS EA does not reproduce the reference -- timing invalid"

# ---- re-verify the ellipse FEniCS row the same way ----
ELM = os.path.join(HERE, "ellipse", "meshes")
d2 = np.load(os.path.join(OUT, "ex2_ellipse.npz"))
ea_ell_ref = float(d2["iso_thick_solid"][0, 0])   # meshes dir currently holds the thick build
tf_e, df_e, ea_e = t_fenics_2d(os.path.join(ELM, "solid_ell4cell_iso.yaml"), "ellipse_v")
e_ell = 100.0 * (ea_e - ea_ell_ref) / ea_ell_ref
print("ellipse verify: FEniCS %5.2fs (%d dof) EA err %+.2f%% vs JAX-solid ref" % (tf_e, df_e, e_ell))
assert abs(e_ell) < 3.0, "ellipse FEniCS EA mismatch -- timing invalid"

# ---- patch the npz rows ----
d = np.load(os.path.join(OUT, "timing4.npz"), allow_pickle=True)
names = list(d["names"]); rows = d["rows"].copy()
i = names.index("two_cell")
rows[i] = [rows[i][0], rows[i][1], dj, tj1, tj2, df, tf]
j = names.index("ellipse")
rows[j][5], rows[j][6] = df_e, tf_e
np.savez(os.path.join(OUT, "timing4.npz"), names=names, rows=rows)
print("patched timing4.npz two_cell + ellipse rows")
