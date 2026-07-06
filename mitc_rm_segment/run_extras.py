"""run_extras.py -- ablation + locking probe under the ALL-6-DOF pipeline (server)."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import taper_study as ts
import run_indep as ri
import segment_indep

BENCH = os.path.join(REPO, "examples", "data", "benchmark")
orig = segment_indep.assemble_segment_indep

print("=== SEGMENT shear-scheme ablation (all-6DOF pipeline, NC48, thin aR=0.7) ===")
for geom, MD, npzn, mats in [("square", os.path.join(HERE, "out", "taper_square", "meshes"),
                              "taper_square_solid_%s.npz", ["iso", "m45"]),
                             ("circle", os.path.join(HERE, "out", "taper_study", "meshes"),
                              "taper_study_solid_%s.npz", ["m45"])]:
    RES = os.path.join(HERE, "out", "extras_%s" % geom)
    os.makedirs(RES, exist_ok=True)
    for mat in mats:
        b = np.load(os.path.join(BENCH, npzn % mat), allow_pickle=True)
        tg = ts.tag_of("thin", mat, 0.7)
        So = 0.5 * (b[tg + "_seg"] + b[tg + "_seg"].T)
        for sch in ("full", "mitc4_wonly", "mitc4_g23", "mitc4_both"):
            def patched(*a, **k):
                k["shear"] = sch
                return orig(*a, **k)
            segment_indep.assemble_segment_indep = patched
            import importlib
            importlib.reload(ri)
            S = ri.shell_solve_lagrange(tg, MD, RES)
            e = [round(100 * (S[i, i] - So[i, i]) / So[i, i], 1) for i in range(6)]
            print("%s thin %-4s %-11s EA %+5.1f GA2 %+6.1f GA3 %+6.1f GJ %+5.1f EI2 %+4.1f EI3 %+4.1f"
                  % (geom, mat, sch, *e))
segment_indep.assemble_segment_indep = orig
import importlib
importlib.reload(ri)

print("\n=== LOCKING probe (all-6DOF): prismatic iso circle vs analytic, coarse mesh ===")
for tR in (0.02, 0.002):
    ts.THICK["thin"] = tR
    a = ts.ana_iso(1.0, tR)
    for (nc, nl) in ((24, 5), (48, 10)):
        mdir = os.path.join(HERE, "out", "lock6", "c_t%s_nc%d" % (str(tR).replace(".", "p"), nc))
        RES = os.path.join(HERE, "out", "lock6", "res")
        os.makedirs(RES, exist_ok=True)
        ts.gen_case("thin", "iso", 1.0, mesh_dir=mdir, nc=nc, nl=nl)
        S = ri.shell_solve_lagrange(ts.tag_of("thin", "iso", 1.0), mdir, RES)
        e = [round(100 * (S[i, i] - a[i]) / a[i], 2) for i in range(6)]
        print("t/R=%-6s %2dx%-3d  EA %+5.2f  GA2 %+6.2f  GA3 %+6.2f  GJ %+6.2f  EI %+5.2f/%+5.2f"
              % (tR, nc, nl, e[0], e[1], e[2], e[3], e[4], e[5]))
ts.THICK["thin"] = 0.02
