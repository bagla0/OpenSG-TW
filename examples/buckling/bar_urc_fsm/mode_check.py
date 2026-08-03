"""mode_check.py -- the eigenvalue matched (st06 lam=1.047 vs benchmark ~1.04 near st5). Now check the two
things that decide whether that is physics or coincidence:

  1. WHERE on the contour does the mode live?  The benchmark mode is in the SPAR CAP between the webs.
     A right eigenvalue in the wrong place is not a validation.
  2. Is lam converged in M (the harmonic count)?  M sets the shortest resolvable half-wave L/M; if lam is
     still falling at M=12 the reported value is not converged and the match is luck.

Also re-derives the ACTUAL critical half-wave from the dominant harmonic, because blade_fsm's fsm_regime
guard is self-fulfilling (it passes a_crit = L/M against spacing L, giving rho = 1/M -> unconditional PASS).
"""
import os, sys
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

ST = int(os.environ.get("ST", "6"))
shell = os.path.join(SHELLD, "1Dshell_%d.yaml" % ST)
B = dehom_rm.build_rm_bundle(shell, ref="oml")
K = solve_tw_from_yaml(shell, frac=0.0)
nd = np.asarray(B["corners"], float)
cells = np.asarray(B["red_cells"], int); cells -= cells.min()
ABD_e = np.asarray(K["ABD_elems"], float)
names = list(B["layup_per_elem"])
FF = ff_station(ST)
st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
N_e = np.zeros((len(cells), 3))
for e in range(len(cells)):
    s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    N_e[e] = ABD_e[e][:3, :3] @ s6[:3] + ABD_e[e][:3, 3:] @ s6[3:6]

print("BAR-URC station %d   (z=%.2f m, r/R=%.3f)   benchmark ~1.04 near st5, spar cap\n"
      % (ST, ST * DZ, ST / 29.0))

# ---------- 1. convergence in M ----------
print("harmonic convergence:")
print("    M     lam1        L/M [m]   change")
prev = None
for M in (4, 6, 8, 12, 16, 20, 24):
    lam = np.asarray(fsm.solve_fsm_multi(nd, cells, list(ABD_e), list(N_e), DZ, M, n_modes=4))
    lam = lam[np.isfinite(lam)]
    l1 = float(lam[0]) if lam.size else np.inf
    ch = "" if prev is None else "%+.2f%%" % (100 * (l1 / prev - 1))
    print("   %3d   %9.4f   %7.4f   %s" % (M, l1, DZ / M, ch))
    prev = l1

# ---------- 2. where does the mode live? ----------
M = 20
lam, V = fsm.solve_fsm_multi(nd, cells, list(ABD_e), list(N_e), DZ, M, n_modes=4, return_vecs=True)
lam = np.asarray(lam)
V = np.asarray(V)                                    # (M, nn, 4, n_modes)
w = V[:, :, 2, 0]                                    # out-of-plane DOF, first mode, per harmonic
amp_node = np.sqrt((w ** 2).sum(axis=0))             # rms over harmonics -> per-node amplitude
amp_node /= amp_node.max() + 1e-30
amp_el = 0.5 * (amp_node[cells[:, 0]] + amp_node[cells[:, 1]])

# dominant harmonic -> the REAL critical half-wave (not the self-fulfilling L/M)
e_h = (w ** 2).sum(axis=1)
mstar = int(np.argmax(e_h)) + 1
print("\n   lam1=%.4f at M=%d;  dominant harmonic m*=%d  ->  a_crit = L/m* = %.3f m"
      % (lam[0], M, mstar, DZ / mstar))
print("   (half-wave / station spacing = %.3f -- a local buckle should be well under 1)" % (1.0 / mstar))

print("\n   mode amplitude by LAYUP group (rms over harmonics, normalised):")
order = {}
for e, nm in enumerate(names):
    order.setdefault(nm, []).append(e)
rows = []
for nm, idx in order.items():
    idx = np.array(idx)
    rows.append((float(amp_el[idx].mean()), float(amp_el[idx].max()), nm, len(idx),
                 float(np.abs(N_e[idx, 0]).max())))
rows.sort(key=lambda r: -r[1])
print("      layup            n_el   mean_amp   max_amp    |N11|max [N/m]")
for mean_a, max_a, nm, n, nmax in rows:
    bar = "#" * int(28 * max_a)
    print("      %-14s   %3d    %6.3f     %6.3f    %10.3e  %s" % (nm, n, mean_a, max_a, nmax, bar))

top = np.argsort(-amp_el)[:8]
print("\n   top-8 elements by amplitude (element, layup, midpoint y2,y3, N11):")
for e in top:
    mid = 0.5 * (nd[cells[e, 0]] + nd[cells[e, 1]])
    print("      e%3d  %-14s  (%8.4f, %8.4f)   N11=%+.3e   amp=%.3f"
          % (e, names[e], mid[0], mid[1], N_e[e, 0], amp_el[e]))
np.savez(os.path.join(HERE, "mode_st%d.npz" % ST), nodes=nd, cells=cells, amp=amp_el,
         names=np.array(names), N=N_e, lam=lam, mstar=mstar)
