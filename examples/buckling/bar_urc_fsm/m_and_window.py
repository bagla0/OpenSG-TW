"""m_and_window.py -- two practical questions for realistic blades.

Q1. Is M = 25-30 a safe convention, or does it "over-converge"?
    Adding harmonics is Rayleigh-Ritz: the computed eigenvalue decreases MONOTONICALLY toward the true one,
    so a larger M can never be less accurate -- only more expensive.  The real trap is ngauss: the connected
    integrand contains products of up to 2M half-waves, so ngauss must scale with M (rule: ngauss >= 4M).
    Raise M with a FIXED ngauss and the longitudinal quadrature aliases, breaking the orthogonality that the
    prismatic limit depends on.  Test both branches on the prismatic webbed member, where the exact answer is
    known (connected must equal per-station to 1.000000).

Q2. A realistic blade has 50-60 stations.  What does CONNECTED cost, and over what span is it meaningful?
    Connecting the whole blade is neither affordable nor physically necessary: a local buckle of half-wave
    ~0.86 m cannot feel the section 50 m away.  The meaningful domain is a WINDOW of a few half-waves.  Sweep
    the window length (with M scaled to hold L/M roughly constant) and find where lambda stabilizes; that
    window length, not the blade length, is what a production run should use.
    Cost model: ndof = 4*nn*M, dense K and KG each ndof^2*8 bytes, assembly ~ ngauss*ns*M^2.
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
from bar_urc_fsm import DZ, SHELLD, ff_station


def station_data(k):
    """contour, connectivity, per-element OML ABD, dehom pre-buckling N.  Defined here rather than imported
    from bar_urc_connected, whose module body executes the whole earlier study on import."""
    shell = os.path.join(SHELLD, "1Dshell_%d.yaml" % k)
    B = dehom_rm.build_rm_bundle(shell, ref="oml")
    K = solve_tw_from_yaml(shell, frac=0.0)
    nd = np.asarray(B["corners"], float)
    cells = np.asarray(B["red_cells"], int); cells -= cells.min()
    ABD = np.asarray(K["ABD_elems"], float)
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=ff_station(k))
    N = np.zeros((len(cells), 3))
    for e in range(len(cells)):
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        N[e] = ABD[e][:3, :3] @ s6[:3] + ABD[e][:3, 3:] @ s6[3:6]
    return nd, cells, ABD, N

NN = 62
MEM_CAP_GB = 6.0


def mem_gb(M, nn=NN):
    nd = 4 * nn * M
    return 2 * nd * nd * 8 / 1024 ** 3


print("BAR-URC: harmonic-count convention and connected-window feasibility\n")
d6 = station_data(6)
nd6, cells6, ABD6, N6 = d6

# ---------------------------------------------------------------- Q1a: monotonicity of M (per-station)
print("Q1a  per-station lambda vs M  (Rayleigh-Ritz => must be non-increasing)")
print("      M      lam1        change     ndof    dense mem")
prev = None
for M in (4, 8, 12, 16, 20, 25, 30, 40):
    t0 = time.time()
    lam = np.asarray(fsm.solve_fsm_multi(nd6, cells6, list(ABD6), list(N6), DZ, M, n_modes=3))
    lam = lam[np.isfinite(lam)]
    l1 = float(lam[0]) if lam.size else np.inf
    ch = "" if prev is None else "%+.3e" % (l1 - prev)
    print("   %4d   %.6f   %10s   %6d   %6.2f GB   (%.0fs)"
          % (M, l1, ch, 4 * NN * M, mem_gb(M), time.time() - t0))
    prev = l1

# ---------------------------------------------------------------- Q1b: the ngauss trap
print("\nQ1b  the ngauss trap -- prismatic webbed member, connected MUST equal per-station (1.000000)")
print("      M   ngauss   ratio        verdict")
secs_p = [(x, nd6, ABD6, N6) for x in (0.0, 0.5 * DZ, DZ)]      # identical sections => prismatic
for M in (8, 16, 25, 30):
    per = np.asarray(fsm.solve_fsm_multi(nd6, cells6, list(ABD6), list(N6), DZ, M, n_modes=3))
    per = per[np.isfinite(per)][0]
    for ng, tag in ((48, "FIXED 48"), (4 * M, "4M")):
        con = np.asarray(fsm.solve_fsm_connected(secs_p, DZ, M, n_modes=3, ngauss=ng, strips=cells6))
        con = con[np.isfinite(con)]
        r = float(con[0]) / per if con.size else np.nan
        ok = "OK" if abs(r - 1) < 1e-3 else "*** ALIASED ***"
        print("   %4d   %3d (%-8s)  %.6f   %s" % (M, ng, tag, r, ok))

# ---------------------------------------------------------------- Q2: connected window sweep
print("\nQ2  connected over a WINDOW of stations centred on the governing station 6")
print("    (M scaled to hold L/M ~ 0.29 m, the value that converged the single-spacing run)")
print("      window      L [m]    M   ngauss   ndof    mem      lam1        vs per-station   time")
per_ref = None
lam_ref = np.asarray(fsm.solve_fsm_multi(nd6, cells6, list(ABD6), list(N6), DZ, 12, n_modes=3))
per_ref = float(lam_ref[np.isfinite(lam_ref)][0])
print("      st6 alone (per-station, L=%.3f)                                    %.4f   (reference)"
      % (DZ, per_ref))

CACHE = {6: d6}
for (k0, k1) in [(5, 6), (5, 7), (4, 7), (4, 8), (3, 9)]:
    L = (k1 - k0) * DZ
    M = int(np.ceil(L / 0.29))
    if mem_gb(M) > MEM_CAP_GB:
        print("      st%d-%d      %6.3f  %3d   %4d   %6d  %5.2f GB   SKIPPED: over the %.0f GB cap"
              % (k0, k1, L, M, 4 * M, 4 * NN * M, mem_gb(M), MEM_CAP_GB))
        continue
    for k in range(k0, k1 + 1):
        if k not in CACHE:
            CACHE[k] = station_data(k)
    secs = [(float(k - k0) * DZ, CACHE[k][0], CACHE[k][2], CACHE[k][3]) for k in range(k0, k1 + 1)]
    t0 = time.time()
    try:
        lam = np.asarray(fsm.solve_fsm_connected(secs, L, M, n_modes=3, ngauss=4 * M, strips=cells6))
        lam = lam[np.isfinite(lam)]
        l1 = float(lam[0]) if lam.size else np.inf
        print("      st%d-%d      %6.3f  %3d   %4d   %6d  %5.2f GB   %.4f    %8.4f      %5.0fs"
              % (k0, k1, L, M, 4 * M, 4 * NN * M, mem_gb(M), l1, l1 / per_ref, time.time() - t0))
    except Exception as ex:
        print("      st%d-%d      FAILED %s: %s" % (k0, k1, type(ex).__name__, ex))

# ---------------------------------------------------------------- extrapolation to a 50-60 station blade
print("\n  extrapolation -- connecting a WHOLE blade at this resolution:")
for nst, span in ((30, 100.0), (60, 100.0)):
    dz = span / (nst - 1)
    M_full = int(np.ceil(span / 0.29))
    print("    %2d stations, dz=%.2f m: connecting the FULL %.0f m span needs M ~ %d -> ndof %d, dense %.0f GB"
          % (nst, dz, span, M_full, 4 * NN * M_full, mem_gb(M_full)))
print("    => connecting the entire blade is infeasible dense, and unnecessary: a local buckle of half-wave")
print("       0.862 m cannot feel the section tens of metres away.  Use a sliding window (see Q2).")
