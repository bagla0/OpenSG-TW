"""ref_llg_check.py -- INDEPENDENT re-measurement of the claimed _L_lg beta-pairing bug.

Written from scratch (does not import any h1*/h2*/h3* probe script).
Part A: element-level rigid-body energy test, my own construction of the 6 rigid modes.
Part B: full blade M_y(FE)/FF_My at stations 15 and 25, before/after the proposed fix,
        via runtime monkeypatch of shell_buckling._L_lg (production file untouched).
"""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import shell_buckling as sb

CODE_L = sb._L_lg


def FIX_L(T):
    """beta_x = e2.theta, beta_y = -e1.theta"""
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


def rigid_energy(X4, ABD, Gs, Lmap):
    """u^T (L^T Ke L) u for the 6 rigid modes about the element centroid."""
    T, xyl = sb._elem_frame(np.asarray(X4, float), np.arange(4))
    Ke, _ = sb.element_K_KG(xyl, ABD, Gs, np.zeros(3))
    L = Lmap(T)
    Kg = L.T @ Ke @ L
    Xc = np.asarray(X4, float) - np.asarray(X4, float).mean(0)
    out = []
    for d in range(3):                      # translations
        ug = np.zeros(24)
        for a in range(4):
            ug[6 * a + d] = 1.0
        out.append(ug @ Kg @ ug / (np.abs(Kg).max() * (ug @ ug)))
    for d in range(3):                      # rotations
        th = np.eye(3)[d]; ug = np.zeros(24)
        for a in range(4):
            ug[6 * a:6 * a + 3] = np.cross(th, Xc[a]); ug[6 * a + 3:6 * a + 6] = th
        out.append(ug @ Kg @ ug / (np.abs(Kg).max() * (ug @ ug)))
    return np.array(out)


print("=" * 78)
print("PART A -- element rigid-body energy (normalized by |Kg|max * u.u); 0 = correct")
print("=" * 78)
ABD, Gs = sb._iso_ABD(70e9, 0.33, 0.01)
cases = {
    "flat unit square": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    "blade-like strip ": [[0, 0, 0], [2.0, 0.05, 0.0], [2.0, 0.9, 0.3], [0.0, 0.8, 0.25]],
    "tilted random    ": None,
}
rng = np.random.default_rng(0)
v1 = rng.normal(size=3); v2 = rng.normal(size=3)
cases["tilted random    "] = [[0, 0, 0], list(v1), list(v1 + v2), list(v2)]
for nm, X in cases.items():
    ec = rigid_energy(X, ABD, Gs, CODE_L)
    ef = rigid_energy(X, ABD, Gs, FIX_L)
    print("  %s" % nm)
    print("     CODE  trans %s  rot %s" % (np.array2string(ec[:3], precision=2),
                                           np.array2string(ec[3:], precision=4)))
    print("     FIX   trans %s  rot %s" % (np.array2string(ef[:3], precision=2),
                                           np.array2string(ef[3:], precision=4)))

print()
print("=" * 78)
print("PART B -- full blade section moment equilibrium")
print("=" * 78)
import blade_iso as bi
import blade_buckling as bb

NSE = bi.NSE; MPER = bb.MPER
bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
print("mesh %d nodes %d quads;  total applied Fz = %.5e" % (len(nodes), len(quads),
                                                            f.reshape(-1, 6)[:, 2].sum()))


def My_from_N(Ne, p, i):
    P = bl["Pk"][i]; M = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zmid = 0.5 * (P[a, 1] + P[b, 1])
        M += -Ne[p * NSE + se, 0] * zmid * ds
    return M


def reaction_moment(u, K):
    r = (K @ u).reshape(-1, 6)
    rr = np.zeros(6, bool); rr[np.asarray(root, int) % 6] = True
    idx = np.unique(np.asarray(root, int) // 6)
    M = np.zeros(3)
    for n in idx:
        M += np.cross(nodes[n], r[n, :3]) + r[n, 3:]
    return M


Mapp = np.zeros(3)
fr = f.reshape(-1, 6)
for n in range(len(nodes)):
    Mapp += np.cross(nodes[n], fr[n, :3]) + fr[n, 3:]

Nrm = bi.rm_N(bl, FF)
print("\nRM dehom reference:")
for i in (15, 25):
    p = min(i * MPER, bl["NS"] - 2)
    print("   sta %2d  M_RM/FF = %+.4f" % (i, My_from_N(Nrm, p, i) / FF[i][4]))

rows = []
for tag, Lmap, kdr in [("CODE  kdr=1e-3", CODE_L, 1e-3),
                       ("FIX   kdr=1e-3", FIX_L, 1e-3),
                       ("FIX   kdr=0   ", FIX_L, 0.0)]:
    sb._L_lg = Lmap
    sb._KDR_SCALE = kdr
    K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
    u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
    Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
    r15 = My_from_N(Nf, min(15 * MPER, bl["NS"] - 2), 15) / FF[15][4]
    r25 = My_from_N(Nf, min(25 * MPER, bl["NS"] - 2), 25) / FF[25][4]
    Mr = reaction_moment(u, K)
    tip = np.abs(u.reshape(-1, 6)[:, 2]).max()
    rows.append((tag, r15, r25, Mr[1] / -Mapp[1], tip))
    print("\n%s : M_FE/FF sta15 = %+.4f   sta25 = %+.4f   Mreact_y/-Mapp_y = %.4f   max|uz| = %.4f m"
          % (tag, r15, r25, Mr[1] / -Mapp[1], tip))

sb._L_lg = CODE_L
print("\n" + "=" * 78)
print(" %-16s %10s %10s %10s %10s" % ("case", "sta15", "sta25", "Mreact", "max|uz|"))
for t, a, b, c, d in rows:
    print(" %-16s %10.4f %10.4f %10.4f %10.4f" % (t, a, b, c, d))
