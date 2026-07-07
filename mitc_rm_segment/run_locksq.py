"""run_locksq.py -- WORST-CASE flat-wall locking sweep (server).

Prismatic SQUARE tube (flat walls: rotation coefficients x_{k;alpha} CONSTANT per
element -- the Q4-plate locking configuration), wall thinned 100x, fixed meshes.
Reference-free locking detector: the prismatic identity segment == ring (same
element, same t).  Locking = segment stiffening away from its own ring as t -> 0
at fixed mesh.  Schemes: full integration, flux-tied, canonical MITC (both rows).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import taper_study as ts
import taper_square as tsq
import run_indep as ri
import segment_indep

orig = segment_indep.assemble_segment_indep
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

print("PRISMATIC SQUARE identity test: 100*(seg - ring)/ring per diagonal term")
print("(locking = growth toward +inf as t/R -> 0 at fixed mesh)")
print("%-9s %-7s %-12s" % ("t/R", "mesh", "scheme") + "".join("%8s" % l for l in LBL))
for tR in (0.02, 0.002, 0.0002):
    ts.THICK["thin"] = tR
    for (nc, nl) in ((24, 5), (48, 10)):
        mdir = os.path.join(HERE, "out", "locksq", "t%s_nc%d" % (str(tR).replace(".", "p"), nc))
        RES = os.path.join(HERE, "out", "locksq", "res")
        os.makedirs(RES, exist_ok=True)
        tsq.gen_square_case("thin", "iso", 1.0, mesh_dir=mdir, nc=nc, nl=nl)
        tg = ts.tag_of("thin", "iso", 1.0)
        for sch, name in (("full", "full int."), ("mitc4_wonly", "flux-tied"),
                          ("mitc4_both", "canon MITC")):
            def patched(*a, **k):
                k["shear"] = sch
                return orig(*a, **k)
            segment_indep.assemble_segment_indep = patched
            import importlib
            importlib.reload(ri)
            r = ri.shell_solve_lagrange(tg, mdir, RES, return_full=True)
            S, CL = r["S6"], r["C6L"]
            e = [100 * (S[i, i] - CL[i, i]) / CL[i, i] for i in range(6)]
            print("%-9s %2dx%-4d %-12s" % (tR, nc, nl, name)
                  + "".join("%+7.2f " % v for v in e))
            sys.stdout.flush()
segment_indep.assemble_segment_indep = orig
ts.THICK["thin"] = 0.02
