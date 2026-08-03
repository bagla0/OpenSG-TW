"""m_rule.py -- is "M = 3 x number of stations" the right convention?

The proposal ties M to the STATION COUNT.  But M does not resolve stations; it resolves the buckle
WAVELENGTH.  What matters is L/M against the critical half-wave a_crit, and L grows with the window while
a_crit is set by the panel (0.862 m here, from m*=4 over one 3.448 m spacing).  So for a window of n spacings:

     m* = L / a_crit = n * 3.448 / 0.862 = 4n          (critical harmonic index)
     "3 x stations"  = 3(n+1) ~ 3n                      (the proposed rule)

i.e. the proposed rule grows at 3n while the requirement grows at 4n *before any margin*.  It therefore
crosses below the critical harmonic as the window grows, and once M < m* the basis cannot represent the
critical mode at all.  Tabulate that crossover, and verify the required margin numerically by sweeping M on a
window whose m* is known.

Also answer whether M=100 is affordable: the current assembly is DENSE (M*4*nn)^2, but K and KG are sparse in
node space within each harmonic-pair block, so the dense storage is an implementation limit, not a
formulation one.
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

NN, NS = 62, 64
A_CRIT = 0.862                      # measured half-wave at the governing station (m*=4 over L=3.448 m)


def station_data(k):
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


print("Does M = 3 x (number of stations) resolve the critical harmonic?\n")
print("  window   stations   L [m]    m* = L/a_crit   M=3xstations   verdict")
for n in range(1, 12):
    L = n * DZ
    mstar = L / A_CRIT
    Mrule = 3 * (n + 1)
    if Mrule < mstar:
        v = "FAILS: below m*, the critical mode is NOT in the basis"
    elif Mrule < 2 * mstar:
        v = "marginal: above m* but under the ~2x margin measured below"
    else:
        v = "adequate"
    print("   %2d sp     %2d      %6.2f       %6.1f          %3d        %s" % (n, n + 1, L, mstar, Mrule, v))

print("\n  -> the rule grows at 3n while the requirement grows at 4n, so it crosses below m* as the")
print("     window grows.  M must scale with L/a_crit, NOT with the station count.")
print("     (nsec, the number of supplied cross-sections, is a SEPARATE convergence parameter: it resolves")
print("      the spanwise VARIATION of the section, not the wavelength of the buckle.)")

# ---------------------------------------------------------------- measured margin on a window with known m*
K0, K1 = 5, 7
L = (K1 - K0) * DZ
D = {k: station_data(k) for k in range(K0, K1 + 1)}
secs = [(float(k - K0) * DZ, D[k][0], D[k][2], D[k][3]) for k in range(K0, K1 + 1)]
cells = D[6][1]
print("\n  measured: connected over stations %d-%d, L=%.4f m, so m* ~ %.0f" % (K0, K1, L, L / A_CRIT))
print("      M    ngauss   lam1       vs M=24    m*_actual   time")
ref = None
for M in (4, 6, 8, 9, 12, 16, 24):
    t0 = time.time()
    out = fsm.solve_fsm_connected(secs, L, M, n_modes=3, ngauss=max(48, 4 * M),
                                  strips=cells, return_vecs=True)
    lam, V = out
    lam = np.asarray(lam); V = np.asarray(V)
    l1 = float(lam[np.isfinite(lam)][0])
    e_h = (V[:, :, 2, 0] ** 2).sum(axis=1)
    ms = int(np.argmax(e_h)) + 1
    if M == 24:
        ref = l1
    print("   %4d   %5d   %.6f   %9s   %6d      %5.0fs"
          % (M, max(48, 4 * M), l1, "-" if ref is None else "", ms, time.time() - t0))
    if M == 9:
        print("            ^ this is the proposed 3 x stations rule for a 3-station window")
if ref is not None:
    print("   (reference M=24 lam = %.6f)" % ref)

# ---------------------------------------------------------------- is M=100 affordable?
print("\n  cost of large M (nn=%d, ns=%d):" % (NN, NS))
print("      M     ndof     DENSE K+KG    est. SPARSE K+KG   dense/sparse")
for M in (12, 25, 30, 50, 100, 200):
    ndof = 4 * NN * M
    dense = 2 * ndof * ndof * 8 / 1024 ** 3
    nnz = NS * 64 * M * M                      # 8x8 strip block per harmonic pair
    sparse = 2 * nnz * 12 / 1024 ** 3          # ~8 B value + 4 B index
    print("   %4d   %7d   %8.2f GB    %10.2f GB      %6.1fx" % (M, ndof, dense, sparse, dense / sparse))
print("   -> M=100 is ~10 GB dense but under 1 GB sparse. The dense (M*4*nn)^2 assembly is an")
print("      IMPLEMENTATION limit, not a formulation one: within each harmonic-pair block the matrix is")
print("      sparse in node space (each strip touches 8 DOF). Moving to scipy.sparse + shift-invert Lanczos")
print("      would make M=100 routine. That is the change to make if high M is wanted for accuracy.")
