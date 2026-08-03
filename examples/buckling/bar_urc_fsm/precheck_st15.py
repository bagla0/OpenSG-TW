"""precheck_st15.py -- decompose the RM-vs-VABS st15 dehom error before trusting N for the FSM.

dehom_st15_paths.py reports a single Frobenius ratio: circumferential 29.3%, cap-centre 26.8%, yet the
PEAK |S11| agrees to 0.6% / 0.4%.  A distributed 29% with a matching peak is either (a) a handful of bad
points at web/cap junctions dominating the norm, or (b) a systematic error in the small components
(S22, S12) that the S11-dominated peak hides.  Those have opposite consequences for the FSM: (a) is
cosmetic for N11-driven spar-cap buckling, (b) is not.

So: report PER-COMPONENT and PER-POINT, with medians and percentiles rather than one norm.
The FSM's pre-buckling state is N = A eps + B kappa, dominated by S11, so the S11 column is the one
that actually gates the buckling result.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml, stress_at_points

DEH = os.path.join(ROOT, "examples", "data", "dehom_st15")
SHELL15 = os.path.join(ROOT, "tests", "data", "1Dshell_15.yaml")
SM = os.path.join(DEH, "bar_urc-15-t-0.in.SM")
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
# VABS order [F1,F2,F3,M1,M2,M3] -- matches the .glb decode (line5 = F1 M1 M2 M3, line6 = F2 F3)
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
PATHS = [("circumferential", "solid.circumferential_015.coords"),
         ("cap-centre", "solid.lp_sparcap_center_thickness_015.coords")]


def load_sm(path):
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]


sm_xy, sm_s = load_sm(SM)
tree = cKDTree(sm_xy)
bundle = solve_tw_from_yaml(SHELL15, frac=0.0)          # frac=0 -> OML reference
print("st15 RM dehom vs VABS .SM   (FF from bar_urc-15-t-0.in.glb, OML ref)\n")

for name, fn in PATHS:
    p = os.path.join(DEH, fn)
    coords = np.loadtxt(p)[:, :2]
    S = np.asarray(stress_at_points(bundle, coords, beam_force_vabs=FF, frame="material")["stress"])
    dist, idx = tree.query(coords)
    V = sm_s[idx]
    print("=" * 92)
    print("%s   npts=%d   nearest-.SM-point distance: med=%.4f mm  max=%.4f mm"
          % (name, len(coords), 1e3 * np.median(dist), 1e3 * dist.max()))

    # ---- per COMPONENT: is the error concentrated in the small components? ----
    print("\n   comp     ||V|| [MPa]   rel_err     med|dS| [MPa]   max|dS| [MPa]   share of total err^2")
    tot = np.sum((S[:, :3].tolist() and (S - V)[:, [0, 1, 5]]) ** 2)
    for c in (0, 1, 5):
        dv = S[:, c] - V[:, c]
        rel = np.linalg.norm(dv) / (np.linalg.norm(V[:, c]) + 1e-30) * 100
        print("   %-6s  %10.2f   %7.1f%%   %12.3f   %12.3f   %14.1f%%"
              % (COMP[c], np.linalg.norm(V[:, c]) / 1e6, rel, np.median(np.abs(dv)) / 1e6,
                 np.abs(dv).max() / 1e6, 100 * np.sum(dv ** 2) / (tot + 1e-30)))

    # ---- per POINT on S11: a few junction spikes, or everywhere? ----
    d11 = np.abs(S[:, 0] - V[:, 0])
    scale = np.abs(V[:, 0]).max() + 1e-30
    pe = 100 * d11 / scale                                  # error as % of the path's peak |S11|
    print("\n   S11 per-point error as %% of peak |S11| (%.1f MPa):" % (scale / 1e6))
    print("      median=%.2f%%   p75=%.2f%%   p90=%.2f%%   p95=%.2f%%   max=%.2f%%"
          % (np.median(pe), np.percentile(pe, 75), np.percentile(pe, 90),
             np.percentile(pe, 95), pe.max()))
    nbad = int((pe > 5).sum())
    print("      points over 5%% of peak: %d / %d  (%.0f%%)" % (nbad, len(pe), 100 * nbad / len(pe)))
    if nbad:
        w = np.argsort(-pe)[:min(6, nbad)]
        print("      worst points (idx, y2, y3, shell MPa, vabs MPa, err%% of peak):")
        for i in w:
            print("         %3d  (%8.4f, %8.4f)  %9.2f  %9.2f   %6.1f%%"
                  % (i, coords[i, 0], coords[i, 1], S[i, 0] / 1e6, V[i, 0] / 1e6, pe[i]))
    # verdict for the FSM: only S11 matters for an N11-driven cap mode
    good = np.median(pe) < 2.0 and nbad <= max(2, int(0.1 * len(pe)))
    print("\n   -> S11 %s for FSM use (median %.2f%% of peak, %d outliers)"
          % ("USABLE" if good else "NOT usable", np.median(pe), nbad))
