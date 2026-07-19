"""h2_equil_identity.py -- the free-body moment identity MUST hold if K u = f.  It didn't (0.85).
So: measure the actual solve residual ||K u - f|| and the conditioning / singularity of K_ff."""
import os, sys, time
import numpy as np
import scipy.sparse as spx
from scipy.sparse.linalg import spsolve, splu
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUCK)
import shell_buckling as sb


def _L_lg_fixed(T):
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


import blade_iso as bi
import blade_buckling as bb
NSE = bi.NSE

for tag, fn in [("ORIG", sb._L_lg), ("FIXED", _L_lg_fixed)]:
    sb._L_lg = fn
    bl = bi.build()
    nodes, quads = bl["nodes"], bl["quads"]
    ABD_e, Gs_e, root = bl["ABD_e"], bl["Gs_e"], bl["root"]
    f = bb.traction_load(nodes, quads)
    FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
    K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), np.asarray(root, int))
    Kff = K[free][:, free].tocsc()
    t0 = time.time()
    uf = spsolve(Kff, f[free])
    r = Kff @ uf - f[free]
    print("\n=== %s ===" % tag)
    print("  spsolve %.0fs   ||K u - f||/||f|| (free) = %.3e   max|r| = %.3e   ||f||=%.3e"
          % (time.time() - t0, np.linalg.norm(r) / np.linalg.norm(f[free]), np.abs(r).max(), np.linalg.norm(f[free])))
    u = np.zeros(ndof); u[free] = uf
    # GLOBAL equilibrium: total reaction at the root vs applied
    R = (K @ u) - f
    print("  root reaction sum F = %s   (applied total F = %s)"
          % (np.array2string(R.reshape(-1, 6)[:, :3].sum(0), precision=4),
             np.array2string(f.reshape(-1, 6)[:, :3].sum(0), precision=4)))
    Rx = R.reshape(-1, 6)
    Mreac = np.cross(nodes, Rx[:, :3]).sum(0) + Rx[:, 3:6].sum(0)
    print("  root reaction moment  = %s   (must cancel FF[0]=%s)"
          % (np.array2string(Mreac, precision=4), np.array2string(FF[0][3:], precision=4)))
    # smallest eigen-ish probe: diagonal scaling & rank hints
    d = Kff.diagonal()
    print("  Kff diag: min=%.3e max=%.3e ratio=%.2e ; zero-rows=%d"
          % (d.min(), d.max(), d.max() / max(d.min(), 1e-300), int((np.abs(Kff).sum(1).A.ravel() == 0).sum())))
    # iterative refinement: does one extra Newton step change the moment ratio?
    lu = splu(Kff)
    uf2 = uf - lu.solve(r)
    u2 = np.zeros(ndof); u2[free] = uf2
    r2 = Kff @ uf2 - f[free]
    print("  after 1 iterative-refinement step: ||r||/||f|| = %.3e" % (np.linalg.norm(r2) / np.linalg.norm(f[free])))
    for uu, lbl in [(u, "raw"), (u2, "refined")]:
        Nf = sb.element_membrane_N(nodes, quads, ABD_e, uu)
        out = []
        for i in [5, 15, 25, 35, 45]:
            p = min(i * bb.MPER, bl["NS"] - 2); P = bl["Pk"][i]; M = 0.0
            for se in range(NSE):
                a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
                ds = np.linalg.norm(P[b] - P[a]); zm = 0.5 * (P[a, 1] + P[b, 1])
                M += -Nf[p * NSE + se, 0] * zm * ds
            out.append(M / FF[i][4])
        print("  [%s] membrane-moment ratios 5/15/25/35/45 = %s   tip=%.3f"
              % (lbl, "  ".join("%7.4f" % v for v in out), uu[6 * int(np.argmax(nodes[:, 0])) + 2]))
