"""h1d_residual_gap.py -- with the _L_lg fix, how much non-invariance is left, and is it the
grounded drilling spring?  Also: full spanwise M_FE/FF profile and the axial-force check."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import blade_iso as bi
import blade_buckling as bb
import shell_buckling as sb

np.set_printoptions(linewidth=230)
NSE = bi.NSE; MPER = bb.MPER; Ntot = bb.Ntot
_L_orig = sb._L_lg


def _L_fixed(T):
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
NS = bl["NS"]; ndof = 6 * len(nodes)
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
fx = f.reshape(-1, 6)[:, :3]
Mapp = np.cross(nodes, fx).sum(0)
free = np.setdiff1d(np.arange(ndof), np.asarray(root, int))


def sec_int(Ne, p, i):
    """returns (-oint N11 z ds, oint N11 ds)"""
    P = bl["Pk"][i]; M = 0.0; F = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zmid = 0.5 * (P[a, 1] + P[b, 1])
        M += -Ne[p * NSE + se, 0] * zmid * ds; F += Ne[p * NSE + se, 0] * ds
    return M, F


def rigid_rot_residual(K, axis):
    """||K u_rigid|| relative -- u_rigid = rigid rotation about `axis` through the origin."""
    th = np.zeros(3); th[axis] = 1.0
    ur = np.zeros(ndof)
    ur[0::6], ur[1::6], ur[2::6] = np.cross(th, nodes).T
    ur[3 + axis::6] = 1.0
    r = K @ ur
    return np.linalg.norm(r[free]) / (np.abs(K).max() * np.linalg.norm(ur[free]) / len(ur[free])**0 + 1e-30), ur @ K @ ur


res = {}
for Llbl, Lf in [("code", _L_orig), ("FIX", _L_fixed)]:
    for kdr in [1e-3, 0.0]:
        sb._L_lg = Lf; sb._KDR_SCALE = kdr
        t0 = time.time()
        K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
        u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
        Rn = (K @ u - f).reshape(-1, 6); rootn = np.arange(Ntot)
        Mr = np.cross(nodes[rootn], Rn[rootn, :3]).sum(0) + Rn[rootn, 3:].sum(0)
        itip = int(np.argmax(nodes[:, 0]))
        Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
        _, Erot = rigid_rot_residual(K, 1)
        # strain energy of the true solution for scale
        Eu = 0.5 * u @ (K @ u)
        print("L=%-4s kdr=%.0e : tip_uz=%8.4f  M_react_y/-M_app_y=%+.4f  E(rigid rot about y)=%.4e  (2*E_sol=%.3e)  (%.0fs)"
              % (Llbl, kdr, u[6 * itip + 2], -Mr[1] / Mapp[1], Erot, 2 * Eu, time.time() - t0))
        res[(Llbl, kdr)] = Nf

print("\n=== spanwise section-moment ratio  M_FE/FF_My   (correct value ~ -1.0, cf. RM dehom -1.02..-1.08) ===")
print("  sta    r      FF_My        code(kdr1e-3)  FIX(kdr1e-3)   FIX(kdr0)   | oint N11 ds  (should be ~0 vs EA scale)")
for i in range(2, 50, 4):
    p = min(i * MPER, NS - 2)
    m_c, F_c = sec_int(res[("code", 1e-3)], p, i)
    m_f, F_f = sec_int(res[("FIX", 1e-3)], p, i)
    m_f0, _ = sec_int(res[("FIX", 0.0)], p, i)
    ff = FF[i][4]
    print("  %3d  %.2f  %+.4e   %+8.4f      %+8.4f     %+8.4f    | code %+.3e  FIX %+.3e"
          % (i, i / 50.0, ff, m_c / ff, m_f / ff, m_f0 / ff, F_c, F_f))
sb._L_lg = _L_orig; sb._KDR_SCALE = 1e-3
