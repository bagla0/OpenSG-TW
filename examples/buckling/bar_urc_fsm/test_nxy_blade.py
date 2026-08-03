"""test_nxy_blade.py -- does the new Nxy geometric term matter where N12 is genuinely nonzero?

The tube A/B showed the term is a no-op under pure axial load, and that is correct physics: a free-ended
tube is traction-controlled, so N11 = -F/(2 pi R) and N22 = N12 = 0 regardless of A16.  N12 only appears
when the section carries shear or torsion.

BAR-URC does: the governing station has M1 = -5.56e4 N.m of torsion plus transverse shear, so the dehom
returns a genuinely nonzero N12 per element.  This is therefore the case that exercises the term.

Reports, per station: max |N12| relative to max |N11| (so the size of the effect is visible), and lambda_1
with and without the term.
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
from bar_urc_fsm import DZ, SHELLD, ff_station


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


STS = [int(v) for v in (sys.argv[1:] or ["5", "6", "7", "8"])]
D = {k: station_data(k) for k in STS}

print("BAR-URC: effect of the Nxy membrane geometric term (N12 is nonzero here -- torsion + shear)\n")
print("   st   max|N11| [N/m]   max|N12| [N/m]   |N12|/|N11|   lam1 OLD     lam1 NEW     change")
for k in STS:
    nd, cells, ABD, N = D[k]
    r11, r12 = np.abs(N[:, 0]).max(), np.abs(N[:, 2]).max()
    vals = {}
    for flag, tag in (("1", "OLD"), ("0", "NEW")):
        os.environ["FSM_NO_NXY"] = flag
        for m in [x for x in list(sys.modules) if "fsm_buckling" in x]:
            del sys.modules[m]
        import fsm_buckling as fsm
        lam = np.asarray(fsm.solve_fsm_multi(nd, cells, list(ABD), list(N), DZ, 12, n_modes=3))
        lam = lam[np.isfinite(lam)]
        vals[tag] = float(lam[0]) if lam.size else np.inf
    ch = 100 * (vals["NEW"] / vals["OLD"] - 1)
    print("   %2d   %13.4e   %13.4e   %10.4f   %9.5f   %9.5f   %+7.3f%%"
          % (k, r11, r12, r12 / r11, vals["OLD"], vals["NEW"], ch))

print("""
   If |N12|/|N11| is small the term should move lambda only slightly; a large shift would indicate the
   geometric stiffness was materially incomplete for a blade.  Either way this is the honest measurement:
   the tube cases could not exercise it at all.
""")
