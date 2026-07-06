"""run_paper_convergence.py -- mesh-convergence + timing study, ALL-6-DOF model.

Sweep (NC, NL) proportionally for the strong-taper (aR=0.7) THIN square and circle,
iso + m45, with the constrained (independent-omega3) element for rings AND segment,
against the fixed 3-D solid reference.  Saves 6x6, %err, and per-stage wall time to
paper_convergence.npz + a human-readable .dat.

    python run_paper_convergence.py
"""
import os, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import taper_study as ts
import taper_square as tsq
import run_indep as ri

BENCH = os.path.join(REPO, "examples", "data", "benchmark")
OUT = os.path.join(HERE, "taper_indep_study")
os.makedirs(OUT, exist_ok=True)
LEVELS = [(24, 5), (36, 8), (48, 10), (72, 15), (96, 20)]   # (NC, NL), coarse -> fine
AR = 0.7


def run():
    res = {}
    log = open(os.path.join(OUT, "paper_convergence.dat"), "w")

    def w(s):
        print(s); sys.stdout.flush(); log.write(s + "\n"); log.flush()

    w("# mesh convergence, ALL-6-DOF model, strong taper aR=0.7, THIN wall (t/R=0.02)")
    w("# %err vs fixed 3-D solid segment reference; times = rings + segment (s)")
    w("# geom mat NC NL  ndof | EA% GA2% GA3% GJ% EI2% EI3% | t_rings t_seg")
    for geom in ("square", "circle"):
        gen_mesh = tsq.gen_square_case if geom == "square" else ts.gen_case
        npzn = "taper_square_solid_%s.npz" if geom == "square" else "taper_study_solid_%s.npz"
        for mat in ("iso", "m45"):
            b = np.load(os.path.join(BENCH, npzn % mat), allow_pickle=True)
            tg = ts.tag_of("thin", mat, AR)
            So = 0.5 * (b[tg + "_seg"] + b[tg + "_seg"].T)
            for (nc, nl) in LEVELS:
                mdir = os.path.join(HERE, "out", "paper_conv", "%s_nc%d_nl%d" % (geom, nc, nl))
                rdir = os.path.join(HERE, "out", "paper_conv", "res_%s" % geom)
                os.makedirs(rdir, exist_ok=True)
                gen_mesh("thin", mat, AR, mesh_dir=mdir, nc=nc, nl=nl)
                key = "%s_%s_nc%d_nl%d" % (geom, mat, nc, nl)
                nnode = nc * (nl + 1)
                try:
                    r = ri.shell_solve_lagrange(tg, mdir, rdir, return_full=True)
                    S = r["S6"]
                    def e(i):
                        return 100 * (S[i, i] - So[i, i]) / So[i, i]
                    w("%s %s %3d %3d %6d | %+7.2f %+7.2f %+7.2f %+7.2f %+7.2f %+7.2f | %6.1f %6.1f"
                      % (geom, mat, nc, nl, 7 * nnode,
                         e(0), e(1), e(2), e(3), e(4), e(5), r["t_rings"], r["t_seg"]))
                    res[key + "_solid"] = So
                    res[key + "_ind"] = S
                    res[key + "_tind"] = r["t_rings"] + r["t_seg"]
                    res[key + "_trings"] = r["t_rings"]
                    res[key + "_tseg"] = r["t_seg"]
                except Exception:
                    w("FAIL %s" % key); traceback.print_exc()
                np.savez(os.path.join(OUT, "paper_convergence.npz"), **res)   # checkpoint
    log.close()
    print("saved", os.path.join(OUT, "paper_convergence.npz"))


if __name__ == "__main__":
    run()
