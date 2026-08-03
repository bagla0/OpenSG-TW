"""bar_urc_connected.py -- CONNECT the BAR-URC stations into one eigenproblem, instead of solving each
station as an isolated prismatic member and taking the minimum.

Why this matters for the paper's claim.  The per-station scan gave st6 lam=1.0468.  That is classical CUFSM:
each cross-section is treated as prismatic, harmonics decouple, and the governing value is min over stations.
It is only exact if the buckle is short compared with the spanwise variation.  The connected model keeps one
harmonic series over the whole run of stations, lets geometry/ABD/N vary with x, and integrates along x
numerically -- so the harmonics couple and the buckle may localize where it likes.

solve_fsm_connected previously hardwired a single closed ring (i, i+1 mod nn) and therefore could not
represent a blade section at all (the three shear webs would have been silently dropped).  It now takes an
optional `strips` argument; this script exercises that path.

Three tests, in order of what they prove:
  1. PRISMATIC WEBBED SANITY -- replicate ONE station at several x.  The member is then prismatic by
     construction but still branched, so connected MUST reproduce solve_fsm_multi.  This validates the new
     branched connected path against a known answer.  Anything other than ~1.000 here invalidates test 3.
  2. Harmonic/quadrature convergence of the connected solve.
  3. The real connected run over the stations spanning the minimum, vs the per-station minimum.
"""
import os, sys, time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (ROOT, os.path.join(ROOT, "examples", "buckling"),
          os.path.join(ROOT, "examples", "TW-paper", "xsec_paper"), HERE):
    sys.path.insert(0, p)
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml
import dehom_rm
import fsm_buckling as fsm
from bar_urc_fsm import ff_station, SHELLD, DZ


def station_data(k):
    """contour, connectivity, per-element OML ABD, and dehom pre-buckling N for station k."""
    shell = os.path.join(SHELLD, "1Dshell_%d.yaml" % k)
    B = dehom_rm.build_rm_bundle(shell, ref="oml")
    K = solve_tw_from_yaml(shell, frac=0.0)
    nd = np.asarray(B["corners"], float)
    cells = np.asarray(B["red_cells"], int); cells -= cells.min()
    ABD = np.asarray(K["ABD_elems"], float)
    FF = ff_station(k)
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
    N = np.zeros((len(cells), 3))
    for e in range(len(cells)):
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        N[e] = ABD[e][:3, :3] @ s6[:3] + ABD[e][:3, 3:] @ s6[3:6]
    return nd, cells, ABD, N


ST0 = int(os.environ.get("ST0", "5"))
ST1 = int(os.environ.get("ST1", "7"))
print("BAR-URC connected FSM   (stations %d..%d, OML ref, flapwise 1200 Pa)\n" % (ST0, ST1))

data = {k: station_data(k) for k in range(ST0, ST1 + 1)}
nd6, cells6, ABD6, N6 = data[min(6, ST1)]

# ---------------------------------------------------------------- 1. prismatic webbed sanity
print("1) PRISMATIC WEBBED SANITY -- one station replicated, connected must equal per-station")
kref = 6 if 6 in data else ST0
ndr, cellsr, ABDr, Nr = data[kref]
Lp = DZ
for M in (6, 10):
    per = np.asarray(fsm.solve_fsm_multi(ndr, cellsr, list(ABDr), list(Nr), Lp, M, n_modes=3))
    per = per[np.isfinite(per)]
    secs = [(x, ndr, ABDr, Nr) for x in (0.0, 0.5 * Lp, Lp)]     # identical sections => prismatic
    con = np.asarray(fsm.solve_fsm_connected(secs, Lp, M, n_modes=3, ngauss=max(48, 4 * M),
                                             strips=cellsr))
    con = con[np.isfinite(con)]
    r = con[0] / per[0] if per.size and con.size else np.nan
    flag = "OK" if abs(r - 1) < 1e-3 else "*** MISMATCH ***"
    print("   M=%2d   per-station=%.6f   connected=%.6f   ratio=%.6f   %s" % (M, per[0], con[0], r, flag))

# ---------------------------------------------------------------- 2/3. the real connected run
span = ST1 - ST0
Lc = span * DZ
print("\n2/3) CONNECTED over stations %d..%d   (L=%.4f m = %d spacings)" % (ST0, ST1, Lc, span))
per_min = np.inf; per_st = None
for k in range(ST0, ST1 + 1):
    ndk, ck, Ak, Nk = data[k]
    lam = np.asarray(fsm.solve_fsm_multi(ndk, ck, list(Ak), list(Nk), DZ, 12, n_modes=3))
    lam = lam[np.isfinite(lam)]
    if lam.size and lam[0] < per_min:
        per_min, per_st = float(lam[0]), k
print("   per-station minimum over this run: st%d  lam=%.4f" % (per_st, per_min))

secs = [(float(k - ST0) * DZ, data[k][0], data[k][2], data[k][3]) for k in range(ST0, ST1 + 1)]
print("\n     M   ngauss   L/M [m]   connected lam1   /per-station-min    time")
for M in (8, 12, 16, 24):
    t0 = time.time()
    try:
        lam = np.asarray(fsm.solve_fsm_connected(secs, Lc, M, n_modes=3, ngauss=max(48, 4 * M),
                                                 strips=cells6))
        lam = lam[np.isfinite(lam)]
        l1 = float(lam[0]) if lam.size else np.inf
        print("   %3d   %5d   %7.4f   %13.4f   %16.4f   %5.0fs"
              % (M, max(48, 4 * M), Lc / M, l1, l1 / per_min, time.time() - t0))
    except Exception as ex:
        print("   %3d   FAILED  %s: %s" % (M, type(ex).__name__, ex))
