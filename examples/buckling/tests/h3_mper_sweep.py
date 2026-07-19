"""h3_mper_sweep.py -- with the CORRECTED _L_lg, is the residual moment deficit just span-mesh
aspect ratio?  Span element length = 138.2/(50*MPER) m while the outboard section elements are
0.02-0.09 m -> aspect ratio 50-100 at MPER=2.  Refine the span and watch M_FE/FF -> -1."""
import os, sys, time, importlib
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import shell_buckling as sb


def _L_lg_fixed(T):
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


sb._L_lg = _L_lg_fixed
import blade_buckling as bb
import blade_iso as bi

for MP in [int(x) for x in (sys.argv[1:] or ["2", "4", "8"])]:
    bb.MPER = MP; bi.MPER = MP
    t0 = time.time(); bl = bi.build()
    nodes, quads = bl["nodes"], bl["quads"]; ABD_e, Gs_e, root = bl["ABD_e"], bl["Gs_e"], bl["root"]
    NS = bl["NS"]; NSE = bi.NSE; itip = int(np.argmax(nodes[:, 0]))
    print("\n##### MPER=%d : %d nodes %d quads %d dof ; span elem = %.3f m"
          % (MP, len(nodes), len(quads), 6 * len(nodes), bb.BLADE_LEN / (50.0 * MP)))
    f = bb.traction_load(nodes, quads)
    FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
    u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root)
    Ne = sb.element_membrane_N(nodes, quads, ABD_e, u)
    print("  tip flap disp = %.4f m   (%.0fs)" % (u[6 * itip + 2], time.time() - t0))
    out = []
    for i in [5, 15, 25, 35, 45]:
        p = min(i * MP, NS - 2); P = bl["Pk"][i]; M = 0.0
        for se in range(NSE):
            a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
            M += -Ne[p * NSE + se, 0] * 0.5 * (P[a, 1] + P[b, 1]) * np.linalg.norm(P[b] - P[a])
        ds = np.linalg.norm(P[bb.sec_elems[:, 1]] - P[bb.sec_elems[:, 0]], axis=1)
        out.append("  sta %2d  M_FE/FF=%8.4f   (sec ds %.3f-%.3f m, aspect %.0f-%.0f)"
                   % (i, M / FF[i][4], ds.min(), ds.max(),
                      bb.BLADE_LEN / (50.0 * MP) / ds.max(), bb.BLADE_LEN / (50.0 * MP) / ds.min()))
    print("\n".join(out))
