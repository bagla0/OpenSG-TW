"""studies_rev.py -- three revision studies for the RM cross-section paper.

A) Transverse-shear constitutive routes: classical FSDT (kappa=5/6 * int Gbar dz),
   Whitney per-direction energy equivalence (coupled=False), and the MSG coupled 2x2
   (coupled=True, complementary energy with full Gbar^{-1}(z)).  Wall G matrices for the
   [-45] ply and beam GA2/GA3 (%err vs 2-D solid) for the m45 cases + IEA r0.2 sandwich.

B) gamma_23 tie regime: 6-DOF ring with shear='mitc4_g23' vs 'full' on iso circular
   tubes across h/R = 0.005..0.2 -- where does full integration lock?

C) 5-DOF (drilling-eliminated MITC, ring_general) vs 6-DOF (Lagrange, ring_indep) on
   the OML IEA rings r0.2/r0.3 -- consistent-convention accuracy + DOF counts.

  -> results/studies_rev.npz  + printed tables
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

from xsec_5v6_master import load_ring, load_solid, ring_6dof, LBL, _norm_materials
from oml_ring import load_ring_ref, c6, derr
from run_ring_indep import ring_indep
from segment_element_general import ring_general
from opensg_jax.fe_jax.msg_transverse_shear import transverse_shear_stiffness, _ply_Q_and_G

TWP = os.path.abspath(os.path.join(HERE, ".."))
TC = os.path.join(TWP, "two_cell_tube", "data")
IB = os.path.join(TWP, "iea22_blade", "data")
ELM = os.path.join(HERE, "ellipse", "meshes")
OUT = os.path.join(HERE, "results")
store = {}


def G_variant(sections, materials, mode):
    """G_by per section: 'msg' | 'whitney' | 'fsdt56'."""
    matmap = {mm["name"]: {"E": mm["elastic"]["E"], "G": mm["elastic"]["G"],
                           "nu": mm["elastic"]["nu"]} for mm in _norm_materials(materials)}
    G_by = {}
    for si, sec in enumerate(sections):
        layup = sec["layup"]
        mn = [p[0] for p in layup]; th = [float(p[1]) for p in layup]; an = [float(p[2]) for p in layup]
        if mode == "msg":
            G_by[si] = np.asarray(transverse_shear_stiffness(th, an, mn, matmap)[0])
        elif mode == "whitney":
            G_by[si] = np.asarray(transverse_shear_stiffness(th, an, mn, matmap, coupled=False)[0])
        elif mode == "fsdt56":
            G = np.zeros((2, 2))
            for k in range(len(th)):
                m = matmap[mn[k]]
                _q1, _q2, gbar = _ply_Q_and_G(m["E"], m["G"], m["nu"], an[k])
                G += np.asarray(gbar) * th[k]
            G_by[si] = (5.0 / 6.0) * G
        else:
            raise ValueError(mode)
    return G_by


def ring_with_G(path, mode, ref=None):
    R = load_ring_ref(path, ref) if ref else load_ring(path)
    d = _yaml.safe_load(open(path))
    R["G_by"] = G_variant(d["sections"], d["materials"], mode)
    return ring_6dof(R) if not ref else c6(R)


print("=" * 88)
print("A) TRANSVERSE-SHEAR CONSTITUTIVE ROUTES  (GA2 | GA3 diag %err vs 2-D solid)")
print("=" * 88)
# wall G for the [-45] single ply, t=4mm, ud-frp (two-cell m45 wall)
m = {"E": [37.0e9, 9.0e9, 9.0e9], "G": [4.0e9] * 3, "nu": [0.28] * 3}
mm = [{"name": "mat", "elastic": m}]
sec1 = [{"layup": [["mat", 0.004, -45.0]]}]
for mode in ("fsdt56", "whitney", "msg"):
    G = G_variant(sec1, mm, mode)[0]
    print("  [-45] wall t=4mm  %-8s G = [[%.4e, %.4e],[%.4e, %.4e]]  (N/m)"
          % (mode, G[0, 0], G[0, 1], G[1, 0], G[1, 1]))
store["wallG_m45"] = np.array([G_variant(sec1, mm, md)[0] for md in ("fsdt56", "whitney", "msg")])

CASES_A = [
    ("two-cell m45", os.path.join(TC, "tube2cell_aniso_thin.yaml"),
     os.path.join(TC, "C6_solid_tube2cell_aniso_thin.txt"), None),
    ("ellipse m45 thin", os.path.join(ELM, "shell_ell4cell_m45.yaml"), None, None),  # solid from npz
    ("IEA r0.2 (OML)", os.path.join(IB, "shell_r020.yaml"),
     os.path.join(IB, "C6_solid_r020.txt"), "oml"),
]
d2 = np.load(os.path.join(OUT, "ex2_ellipse.npz"))
resA = {}
for name, sp, so, ref in CASES_A:
    So = load_solid(so) if so else 0.5 * (d2["m45_thick_solid"] + d2["m45_thick_solid"].T)
    line = [name]
    for mode in ("fsdt56", "whitney", "msg"):
        C = ring_with_G(sp, mode, ref)
        e2 = 100.0 * (C[1, 1] - So[1, 1]) / So[1, 1]
        e3 = 100.0 * (C[2, 2] - So[2, 2]) / So[2, 2]
        line += [e2, e3]
    resA[name] = line[1:]
    print("  %-18s  FSDT(5/6) %+7.2f | %+7.2f   Whitney %+7.2f | %+7.2f   MSG %+7.2f | %+7.2f"
          % tuple([name] + line[1:]), flush=True)
store["A_names"] = [c[0] for c in CASES_A]
store["A_vals"] = np.array([resA[c[0]] for c in CASES_A])

print("=" * 88)
print("B) GAMMA_23 TIE REGIME: 6-DOF tied vs full integration, iso tube R=1, N=200")
print("=" * 88)


class FL(list):
    pass


_yaml.add_representer(FL, lambda d, x: d.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))
E0, NU = 70e9, 0.3
G0 = E0 / (2 * (1 + NU))
SCR = os.path.join(OUT, "_rev_scr"); os.makedirs(SCR, exist_ok=True)


def gen_tube(hR, N=200):
    Rr = 1.0; t = hR * Rr
    nodes = [(Rr * math.cos(2 * math.pi * k / N), Rr * math.sin(2 * math.pi * k / N)) for k in range(N)]
    elems = [(k, (k + 1) % N) for k in range(N)]
    ori = []
    for (a, b) in elems:
        tv = np.array(nodes[b]) - np.array(nodes[a]); e2 = tv / np.linalg.norm(tv)
        e3 = np.array([-e2[1], e2[0]])
        ori.append([0.0, 0.0, 1.0, e2[0], e2[1], 0.0, e3[0], e3[1], 0.0])
    data = {"nodes": [FL(["%.10f %.10f 0.0" % p]) for p in nodes],
            "elements": [FL(["%d %d" % (a + 1, b + 1)]) for (a, b) in elems],
            "sets": {"element": [{"name": "wall", "labels": list(range(1, N + 1))}]},
            "sections": [{"type": "shell", "elementSet": "wall", "layup": [["mat", float(t), 0.0]]}],
            "materials": [{"name": "mat", "density": 2700.0,
                           "elastic": {"E": FL([E0] * 3), "G": FL([G0] * 3), "nu": FL([NU] * 3)}}],
            "elementOrientations": [FL([float(v) for v in o]) for o in ori]}
    p = os.path.join(SCR, "tube_hR%.3f.yaml" % hR)
    _yaml.dump(data, open(p, "w"), sort_keys=False, default_flow_style=False)
    return p, t


rowsB = []
for hR in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
    p, t = gen_tube(hR)
    R = load_ring(p)
    Cs = {}
    for sch in ("mitc4_g23", "full"):
        C = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                       R["k22"], R["ax"], R["cross"], shear=sch, lam_space="elem")
        Cs[sch] = 0.5 * (C + C.T)
    GA_a = G0 * math.pi * 1.0 * t                     # thin-ring analytical GA
    e_t = 100.0 * (Cs["mitc4_g23"][1, 1] - GA_a) / GA_a
    e_f = 100.0 * (Cs["full"][1, 1] - GA_a) / GA_a
    dd = 100.0 * (Cs["full"][1, 1] - Cs["mitc4_g23"][1, 1]) / Cs["mitc4_g23"][1, 1]
    rowsB.append([hR, Cs["mitc4_g23"][1, 1], Cs["full"][1, 1], GA_a, e_t, e_f, dd])
    print("  h/R=%.3f  GA2 tied=%.4e  full=%.4e  ring-analytic=%.4e | err tied %+6.2f%%  "
          "full %+6.2f%% | full-vs-tied %+7.3f%%" % (hR, Cs["mitc4_g23"][1, 1], Cs["full"][1, 1],
                                                     GA_a, e_t, e_f, dd), flush=True)
store["B_rows"] = np.array(rowsB)

print("=" * 88)
print("C) 5-DOF (eliminated MITC) vs 6-DOF (Lagrange) -- OML IEA rings, consistent convention")
print("=" * 88)
for tag in ("r020", "r030"):
    So = load_solid(os.path.join(IB, "C6_solid_%s.txt" % tag))
    R = load_ring_ref(os.path.join(IB, "shell_%s.yaml" % tag), "oml")
    C6m = c6(R)
    C5, _v0, _v1 = ring_general(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                                R["k22"], R["ax"], R["cross"], shear="mitc4_g23")
    C5 = 0.5 * (C5 + C5.T)
    n = len(R["rx"]); ne = len(R["cells"])
    print("  %s: DOFs 5-DOF=%d  6-DOF+lam=%d (+%.0f%%)" % (tag, 5 * n, 6 * n + ne,
          100.0 * (6 * n + ne - 5 * n) / (5 * n)))
    for nm, C in (("6-DOF", C6m), ("5-DOF", C5)):
        e = derr(C, So)
        print("    %-6s diag %%err: %s" % (nm, "  ".join("%s%+6.2f" % (LBL[i], e[i]) for i in range(6))), flush=True)
    store["C_%s_c5" % tag] = C5; store["C_%s_c6" % tag] = C6m; store["C_%s_solid" % tag] = So
    store["C_%s_dofs" % tag] = np.array([5 * n, 6 * n + ne])

np.savez(os.path.join(OUT, "studies_rev.npz"), **store)
print("wrote studies_rev.npz")
