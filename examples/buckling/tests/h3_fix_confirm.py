"""h3_fix_confirm.py -- confirm the root cause: _L_lg maps the global rotation VECTOR onto the local
Mindlin section rotations (beta_x, beta_y) as (e1.r, e2.r), but the element defines
    u = u0 + z*beta_x , v = v0 + z*beta_y , gamma_xz = w,x + beta_x , gamma_yz = w,y + beta_y
so the correct map is  beta_x = e2.r , beta_y = -e1.r  (a 90 deg swap + sign).
The wrong map makes gamma != 0 under a rigid rotation -> the shell shear-locks and carries bending as
spurious transverse shear instead of membrane N.

Monkeypatch the corrected _L_lg and re-run: plate benchmark, SS cylinder benchmark, blade equilibrium.
NOTHING in production code is modified."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import shell_buckling as sb

_L_lg_orig = sb._L_lg


def _L_lg_fixed(T):
    """local-20 (u v w beta_x beta_y) <- global-24 (ux uy uz rx ry rz).
    beta_x = e2.r, beta_y = -e1.r  (so a rigid rotation gives gamma = 0)."""
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"

if WHICH in ("bench", "all"):
    for tag, Lfn in [("ORIGINAL", _L_lg_orig), ("FIXED", _L_lg_fixed)]:
        sb._L_lg = Lfn
        print("\n########## %s _L_lg ##########" % tag)
        sb.validate_plate()
        sb.validate_cylinder(mesh=(160, 80), bc="SS")

if WHICH in ("blade", "all"):
    import blade_iso as bi
    import blade_buckling as bb
    NSE = bi.NSE; MPER = bb.MPER; Ntot = bb.Ntot; NSTA = bb.NSTA
    bl = bi.build()
    nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
    NS = bl["NS"]; itip = int(np.argmax(nodes[:, 0]))
    f_tr = bb.traction_load(nodes, quads)
    FF = bb.beam_forces_from_traction(nodes, f_tr, bl["Rk"])
    FTIP = 1.0e6; tip0 = (NS - 1) * Ntot
    f_tip = np.zeros(6 * len(nodes))
    for n_ in range(tip0, tip0 + Ntot):
        f_tip[6 * n_ + 2] += FTIP / Ntot
    FFt = bb.beam_forces_from_traction(nodes, f_tip, bl["Rk"])

    def moment_from_N(Ne, p, i):
        P = bl["Pk"][i]; M = 0.0
        for se in range(NSE):
            a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
            M += -Ne[p * NSE + se, 0] * 0.5 * (P[a, 1] + P[b, 1]) * np.linalg.norm(P[b] - P[a])
        return M

    def esplit(u):
        Em = Eb = Es = 0.0
        for e, q in enumerate(quads):
            T_, xyl_ = sb._elem_frame(nodes, q)
            ul = sb._L_lg(T_) @ np.concatenate([u[6 * n_:6 * n_ + 6] for n_ in q])
            A = ABD_e[e][:3, :3]; D = ABD_e[e][3:, 3:]; Gs = Gs_e[e]
            _, _, BsA, _, _, JA, _ = sb._B_at(xyl_, 0.0, -1.0); gxiA = (JA @ BsA)[0]
            _, _, BsC, _, _, JC, _ = sb._B_at(xyl_, 0.0, 1.0); gxiC = (JC @ BsC)[0]
            _, _, BsB, _, _, JB, _ = sb._B_at(xyl_, 1.0, 0.0); getB = (JB @ BsB)[1]
            _, _, BsD, _, _, JD, _ = sb._B_at(xyl_, -1.0, 0.0); getD = (JD @ BsD)[1]
            for xi in sb.G2:
                for eta in sb.G2:
                    Bm, Bb_, _, _, detJ, Jg, _ = sb._B_at(xyl_, xi, eta)
                    em = Bm @ ul; kb = Bb_ @ ul
                    Em += em @ A @ em * detJ; Eb += kb @ D @ kb * detJ
                    Bsc = np.linalg.inv(Jg) @ np.vstack([0.5 * (1 - eta) * gxiA + 0.5 * (1 + eta) * gxiC,
                                                         0.5 * (1 + xi) * getB + 0.5 * (1 - xi) * getD])
                    gs = Bsc @ ul; Es += gs @ Gs @ gs * detJ
        t = Em + Eb + Es
        return 100 * Em / t, 100 * Eb / t, 100 * Es / t

    for tag, Lfn in [("ORIGINAL", _L_lg_orig), ("FIXED", _L_lg_fixed)]:
        sb._L_lg = Lfn
        print("\n########## BLADE with %s _L_lg ##########" % tag)
        t1 = time.time(); K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
        u_tr = sb.solve_static(nodes, quads, ABD_e, Gs_e, f_tr, root, K=K)
        u_tip = sb.solve_static(nodes, quads, ABD_e, Gs_e, f_tip, root, K=K)
        N_tr = sb.element_membrane_N(nodes, quads, ABD_e, u_tr)
        N_tip = sb.element_membrane_N(nodes, quads, ABD_e, u_tip)
        print("  tip disp: traction=%.5f m   tip-load=%.5f m   (%.0fs)"
              % (u_tr[6 * itip + 2], u_tip[6 * itip + 2], time.time() - t1))
        print("  energy split traction: membrane %.2f%% bending %.2f%% shear %.2f%%" % esplit(u_tr))
        print("  energy split tip-load: membrane %.2f%% bending %.2f%% shear %.2f%%" % esplit(u_tip))
        print("   sta   FF_My(trac)    M_FE/FF(trac)  |  FF_My(tip)     M_FE/FF(tip)")
        for i in [5, 15, 25, 35, 45]:
            p = min(i * MPER, NS - 2)
            print("   %2d   %+.4e   %10.4f     |  %+.4e   %10.4f"
                  % (i, FF[i][4], moment_from_N(N_tr, p, i) / FF[i][4],
                     FFt[i][4], moment_from_N(N_tip, p, i) / FFt[i][4]))
    sb._L_lg = _L_lg_orig
