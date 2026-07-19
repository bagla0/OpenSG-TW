"""h3_full_moment.py -- with the CORRECTED _L_lg, is the residual moment deficit real, or is the
arbiter integral incomplete?  The beam moment carried by a shell wall is

    M_y = -oint [ N11 * z_mid  +  M11 * n_z ] ds        (n = wall normal, z_mid = mid-surface flap coord)

because sigma_11 acts at z_global = z_mid + zeta*n_z through the wall thickness.  dbg_blade_equil uses
only the N11 term.  Measure BOTH terms.  Also splits the N11 term by skin vs web."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import shell_buckling as sb

ORIG = sb._L_lg
FIX = bool(int(os.environ.get("FIXL", "1")))


def _L_lg_fixed(T):
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


if FIX:
    sb._L_lg = _L_lg_fixed
import blade_buckling as bb
import blade_iso as bi

bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
NS = bl["NS"]; NSE = bi.NSE; MPER = bb.MPER; Ntot = bb.Ntot
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
FTIP = 1.0e6; tip0 = (NS - 1) * Ntot
f_tip = np.zeros(6 * len(nodes))
for n_ in range(tip0, tip0 + Ntot):
    f_tip[6 * n_ + 2] += FTIP / Ntot
FFt = bb.beam_forces_from_traction(nodes, f_tip, bl["Rk"])
K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)


def NM_elem(u):
    """per-element (N11,N22,N12, M11,M22,M12) local + element normal n (global)."""
    NN = np.zeros((len(quads), 6)); nrm = np.zeros((len(quads), 3))
    for e, q in enumerate(quads):
        T, xyl = sb._elem_frame(nodes, q); nrm[e] = T[2]
        ul = sb._L_lg(T) @ np.concatenate([u[6 * n_:6 * n_ + 6] for n_ in q])
        eps = np.zeros(3); kap = np.zeros(3); ar = 0.0
        for xi in sb.G2:
            for eta in sb.G2:
                Bm, Bb_, _, _, detJ, _, _ = sb._B_at(xyl, xi, eta)
                eps += (Bm @ ul) * detJ; kap += (Bb_ @ ul) * detJ; ar += detJ
        eps /= ar; kap /= ar
        NN[e, :3] = ABD_e[e][:3, :3] @ eps + ABD_e[e][:3, 3:] @ kap
        NN[e, 3:] = ABD_e[e][3:, :3] @ eps + ABD_e[e][3:, 3:] @ kap
    return NN, nrm


def report(tag, u, FFx):
    NN, nrm = NM_elem(u)
    print("\n--- %s ---" % tag)
    print(" sta   FF_My        -oint N11 z ds   -oint M11 nz ds     TOTAL      TOT/FF   (N-only)/FF   skin/web N-split")
    for i in [5, 15, 25, 35, 45]:
        p = min(i * MPER, NS - 2); P = bl["Pk"][i]
        mN = mM = mNs = mNw = 0.0
        for se in range(NSE):
            a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
            ds = np.linalg.norm(P[b] - P[a]); zm = 0.5 * (P[a, 1] + P[b, 1])
            e = p * NSE + se
            t1 = -NN[e, 0] * zm * ds; t2 = -NN[e, 3] * nrm[e, 2] * ds
            mN += t1; mM += t2
            if bb.is_web[se]:
                mNw += t1
            else:
                mNs += t1
        tot = mN + mM; ff = FFx[i][4]
        print("  %2d  %+.3e   %+.3e     %+.3e    %+.3e  %7.4f   %7.4f      %+.2e / %+.2e"
              % (i, ff, mN, mM, tot, tot / ff, mN / ff, mNs, mNw))


u_tr = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
u_tip = sb.solve_static(nodes, quads, ABD_e, Gs_e, f_tip, root, K=K)
print("FIXED _L_lg = %s ; tip disp traction=%.4f tip-load=%.4f" % (FIX, u_tr[6 * np.argmax(nodes[:, 0]) + 2],
                                                                  u_tip[6 * int(np.argmax(nodes[:, 0])) + 2]))
report("traction 1500 Pa", u_tr, FF)
report("pure tip load 1e6 N", u_tip, FFt)

# where does the FE axial stress sit?  compare the FE N11 distribution to the RM one at one station
print("\n--- N11 distribution sanity at station 15 (traction) ---")
NN, nrm = NM_elem(u_tr)
p = min(15 * MPER, NS - 2)
n11 = NN[p * NSE:(p + 1) * NSE, 0]
print("  FE N11: min=%+.3e max=%+.3e  |  skin rms=%.3e web rms=%.3e"
      % (n11.min(), n11.max(), np.sqrt((n11[~bb.is_web] ** 2).mean()), np.sqrt((n11[bb.is_web] ** 2).mean())))
