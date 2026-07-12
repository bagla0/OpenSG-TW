"""studies_rev2.py -- follow-ups:
A2) WALL-level transverse-shear G for the actual IEA r0.2 laminates (spar cap, sandwich
    panel, web): FSDT(5/6) vs Whitney vs MSG-coupled.  The sandwich is where FSDT's
    kappa*int(G dz) fails (skin G*t wrongly stiffens the core-dominated compliance).
A2b) beam GA2/GA3 sensitivity: scale the wall G by the FSDT/MSG ratio worst-case?  --
    skipped; beam-level result already in studies_rev.
B2) gamma_23 tie vs FULL integration on the FLAT-WALLED sections across thickness:
    webbed two-cell tube at t/R = 0.008, 0.02, 0.08, 0.32 and the IEA r0.2 ring.
    Reports the max |full-vs-tied| over the 6 diagonal terms -- where does the tie act?
"""
import math
import os
import sys

import numpy as np
import yaml as _yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for q in (HERE, MITC, REPO):
    sys.path.insert(0, q)

from xsec_5v6_master import load_ring, LBL
from oml_ring import load_ring_ref
from run_ring_indep import ring_indep
from studies_rev import G_variant

TWP = os.path.abspath(os.path.join(HERE, ".."))
IB = os.path.join(TWP, "iea22_blade", "data")
OUT = os.path.join(HERE, "results")
SCR = os.path.join(OUT, "_rev_scr"); os.makedirs(SCR, exist_ok=True)

print("=" * 96)
print("A2) WALL-level G (2x2, N/m) for the IEA r0.2 laminates: FSDT(5/6) | Whitney | MSG")
print("=" * 96)
d = _yaml.safe_load(open(os.path.join(IB, "shell_r020.yaml")))
secs, mats = d["sections"], d["materials"]
res = {}
for si, sec in enumerate(secs):
    name = sec.get("elementSet", "sec%d" % si)
    t = sum(float(p[1]) for p in sec["layup"])
    Gs = {md: G_variant([sec], mats, md)[0] for md in ("fsdt56", "whitney", "msg")}
    r11 = Gs["fsdt56"][0, 0] / Gs["msg"][0, 0]
    r22 = Gs["fsdt56"][1, 1] / Gs["msg"][1, 1]
    res[name] = (t, Gs, r11, r22)
    print("  %-22s t=%6.1f mm | G11: F %.3e  W %.3e  M %.3e (F/M=%5.2f) | "
          "G22: F %.3e  M %.3e (F/M=%5.2f)"
          % (name, 1e3 * t, Gs["fsdt56"][0, 0], Gs["whitney"][0, 0], Gs["msg"][0, 0], r11,
             Gs["fsdt56"][1, 1], Gs["msg"][1, 1], r22), flush=True)

print("=" * 96)
print("B2) gamma_23 tie vs FULL integration on FLAT-WALLED sections (max |full-tied| diag %)")
print("=" * 96)


class FL(list):
    pass


_yaml.add_representer(FL, lambda dd, x: dd.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))
E0, NU = 68.9e9, 0.33


def gen_2cell(T, N=200):
    R = 0.05
    NW = max(8, N // 5)
    ang = [2.0 * math.pi * k / N for k in range(N)]
    nodes = [(R * math.cos(a), R * math.sin(a)) for a in ang]
    elems = [(k, (k + 1) % N) for k in range(N)]
    it, ib = N // 4, 3 * N // 4
    pb, pt = np.array(nodes[ib]), np.array(nodes[it])
    prev = ib
    for j in range(1, NW + 1):
        cur = it if j == NW else len(nodes)
        if j != NW:
            nodes.append(tuple(pb + (pt - pb) * j / NW))
        elems.append((prev, cur)); prev = cur
    nodes = np.array(nodes)
    ori = []
    for (a, b) in elems:
        tv = nodes[b] - nodes[a]; e2 = tv / (np.linalg.norm(tv) + 1e-30)
        e3 = np.array([-e2[1], e2[0]])
        ori.append([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0, float(e3[0]), float(e3[1]), 0.0])
    data = {"nodes": [FL(["%.10f %.10f 0.0" % (x, y)]) for (x, y) in nodes],
            "elements": [FL(["%d %d" % (a + 1, b + 1)]) for (a, b) in elems],
            "sets": {"element": [{"name": "wall", "labels": list(range(1, len(elems) + 1))}]},
            "sections": [{"type": "shell", "elementSet": "wall", "layup": [["mat", float(T), 0.0]]}],
            "materials": [{"name": "mat", "density": 2700.0,
                           "elastic": {"E": FL([E0] * 3), "G": FL([E0 / (2 * (1 + NU))] * 3),
                                       "nu": FL([NU] * 3)}}],
            "elementOrientations": [FL([float(v) for v in o]) for o in ori]}
    p = os.path.join(SCR, "twocell_T%.4f.yaml" % T)
    _yaml.dump(data, open(p, "w"), sort_keys=False, default_flow_style=False)
    return p


def tie_vs_full(R):
    Cs = {}
    for sch in ("mitc4_g23", "full"):
        C = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                       R["k22"], R["ax"], R["cross"], shear=sch, lam_space="elem")
        Cs[sch] = 0.5 * (C + C.T)
    dd = [100.0 * (Cs["full"][i, i] - Cs["mitc4_g23"][i, i]) / Cs["mitc4_g23"][i, i]
          for i in range(6)]
    return Cs, dd


for T in (0.0004, 0.001, 0.004, 0.016):
    p = gen_2cell(T)
    _cs, dd = tie_vs_full(load_ring(p))
    print("  two-cell t/R=%.3f : full-vs-tied diag %%: %s   max|.|=%.3f%%"
          % (T / 0.05, "  ".join("%s%+7.3f" % (LBL[i], dd[i]) for i in range(6)),
             max(abs(v) for v in dd)), flush=True)

_cs, dd = tie_vs_full(load_ring_ref(os.path.join(IB, "shell_r020.yaml"), "oml"))
print("  IEA r0.2 (OML)     : full-vs-tied diag %%: %s   max|.|=%.3f%%"
      % ("  ".join("%s%+7.3f" % (LBL[i], dd[i]) for i in range(6)), max(abs(v) for v in dd)), flush=True)
