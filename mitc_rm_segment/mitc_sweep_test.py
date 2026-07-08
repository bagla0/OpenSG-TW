"""mitc_sweep_test.py -- locate the crossover for a 'thin->MITC, thick->full' rule.
Runs the tapered webbed-ellipse SEGMENT shell at t/R in {0.02,0.05,0.10,0.20} for full
2x2 Gauss vs the MITC tie of BOTH transverse-shear strains, comparing the Timoshenko 6x6
diagonal against the conforming 3-D solid.  MITC should reduce the GA3 locking on thin
walls and (harmlessly or harmfully) alias the shear on thick walls -- the sign of that
trade sets the switch point."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))
import run_ell3w as e3w
import run_indep as ri

E = os.path.expanduser("~/claude_tmp/tw6dof/ellipse_prevabs")
SOLID = {0.02: "webhex_thin_solid6.npy", 0.05: "webhex_tR05_solid6.npy",
         0.10: "webhex_tR10_solid6.npy", 0.20: "webhex_thick_solid6.npy"}
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
SCHEMES = ["full", "mitc4_both"]

for T in (0.02, 0.05, 0.10, 0.20):
    e3w.T = T
    tag = "tR%03d" % round(T * 1000)
    OUT = os.path.join(HERE, "out", "mitcsweep_" + tag); os.makedirs(OUT, exist_ok=True)
    shd = os.path.join(OUT, "shell"); e3w.gen_ell3w(shd, nc=48, nl=10, nw=6, nr=4)
    spath = os.path.join(E, SOLID[T])
    sol = np.diag(np.load(spath)) / 1e9 if os.path.exists(spath) else None
    print("\n" + "=" * 78, flush=True)
    print("t/R = %.3g   solid diag = %s" % (T, np.round(sol, 4) if sol is not None else "MISSING"), flush=True)
    print("%-11s %8s %8s %8s %8s %8s %8s" % ("scheme", *LBL), flush=True)
    for sc in SCHEMES:
        try:
            Sh = np.asarray(ri.shell_solve_lagrange("e3w", shd, os.path.join(OUT, "res"), shear=sc))
            d = np.diag(Sh) / 1e9
            print("%-11s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % (sc, *d), flush=True)
            if sol is not None:
                err = 100 * (d - sol) / sol
                print("%-11s %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%%"
                      % ("  %err", *err), flush=True)
            np.save(os.path.join(HERE, "mitcsweep_%s_%s.npy" % (tag, sc)), Sh)
        except Exception as ex:
            print("%-11s FAILED: %s" % (sc, repr(ex)[:140]), flush=True)
print("\ndone", flush=True)
