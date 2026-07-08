"""run_extras.py -- locking probe with the PRODUCTION scheme (server).

Production shear treatment: rings tie gamma_23 only; the tapered segment ties both
rows at the Dvorkin-Bathe points with the rotation columns fully integrated.
Probe: prismatic isotropic circle vs closed-form Timoshenko constants at the
coarsest mesh, thinned tenfold -- thickness-independence of the error = no locking.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import taper_study as ts
import run_indep as ri

print("=== LOCKING probe (production scheme): prismatic iso circle vs analytic ===")
for tR in (0.02, 0.002):
    ts.THICK["thin"] = tR
    a = ts.ana_iso(1.0, tR)
    for (nc, nl) in ((24, 5), (48, 10)):
        mdir = os.path.join(HERE, "out", "lockP", "c_t%s_nc%d" % (str(tR).replace(".", "p"), nc))
        RES = os.path.join(HERE, "out", "lockP", "res")
        os.makedirs(RES, exist_ok=True)
        ts.gen_case("thin", "iso", 1.0, mesh_dir=mdir, nc=nc, nl=nl)
        S = ri.shell_solve_lagrange(ts.tag_of("thin", "iso", 1.0), mdir, RES)
        e = [round(100 * (S[i, i] - a[i]) / a[i], 2) for i in range(6)]
        print("t/R=%-6s %2dx%-3d  EA %+5.2f  GA2 %+6.2f  GA3 %+6.2f  GJ %+6.2f  EI %+5.2f/%+5.2f"
              % (tR, nc, nl, e[0], e[1], e[2], e[3], e[4], e[5]))
ts.THICK["thin"] = 0.02
