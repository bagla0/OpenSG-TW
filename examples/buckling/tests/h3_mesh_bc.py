"""h3_mesh_bc.py -- H3 probe: is the lofted blade mesh / BC defective so it cannot carry beam bending?

(a) edge manifoldness / layer connectivity
(b) duplicate (unmerged) coincident nodes
(c) root BC set = layer 0 = min x
(d) DECISIVE: pure tip transverse load -> does the FE membrane N carry the moment?
(e) strain-energy split (membrane / bending / transverse shear) of the FE static solution
(f) element-level rigid-body-motion patch test of the local<-global DOF transform _L_lg
"""
import os, sys, time
import numpy as np
from scipy.spatial import cKDTree
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import blade_iso as bi
import blade_buckling as bb
import shell_buckling as sb

NSE = bi.NSE; MPER = bb.MPER; NSTA = bb.NSTA; Ntot = bb.Ntot; N = bb.N

t0 = time.time()
bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
NS = bl["NS"]
print("mesh %d nodes %d quads  NS=%d Ntot=%d NSE=%d  (%.1fs)"
      % (len(nodes), len(quads), NS, Ntot, NSE, time.time() - t0))
print("web junction OML nodes: ia=%s ib=%s  (NWEB=%d NWI=%d)" % (bb.web_ia, bb.web_ib, bb.NWEB, bb.NWI))

# ---------------------------------------------------------------- (a) connectivity / manifoldness
print("\n=== (a) EDGE MANIFOLDNESS ===")
from collections import defaultdict
ecount = defaultdict(int)
for q in quads:
    for k in range(4):
        a, b = int(q[k]), int(q[(k + 1) % 4])
        ecount[(min(a, b), max(a, b))] += 1
cnt = np.array(list(ecount.values()))
print("  unique edges = %d ; multiplicity histogram: %s"
      % (len(cnt), {int(v): int((cnt == v).sum()) for v in np.unique(cnt)}))
b_edges = [e for e, c in ecount.items() if c == 1]
nm_edges = [e for e, c in ecount.items() if c > 2]
# classify: span edge (same section node, adjacent planes) vs section edge (same plane)
def cls(e):
    a, b = e
    pa, sa = a // Ntot, a % Ntot; pb, sbx = b // Ntot, b % Ntot
    if sa == sbx and abs(pa - pb) == 1:
        return "span", sa, min(pa, pb)
    if pa == pb:
        return "sect", pa, (sa, sbx)
    return "OTHER", None, None
bs = defaultdict(int)
for e in b_edges:
    bs[cls(e)[0]] += 1
print("  boundary (1-quad) edges = %d  by class %s" % (len(b_edges), dict(bs)))
bp = sorted(set(cls(e)[1] for e in b_edges if cls(e)[0] == "sect"))
print("     -> boundary section edges live on planes %s (expect [0, %d] only); count=%d (expect 2*NSE=%d)"
      % (bp, NS - 1, len([e for e in b_edges if cls(e)[0] == "sect"]), 2 * NSE))
ns = defaultdict(int); nm_secnodes = set()
for e in nm_edges:
    c, sn, _ = cls(e); ns[c] += 1
    if c == "span":
        nm_secnodes.add(sn)
print("  non-manifold (>2 quad) edges = %d  by class %s ; multiplicities %s"
      % (len(nm_edges), dict(ns), sorted(set(ecount[e] for e in nm_edges))))
print("     -> section nodes carrying them: %s" % sorted(nm_secnodes))
print("     -> expected T-junction nodes (web ends): %s" % sorted(set(bb.web_ia) | set(bb.web_ib)))
print("     -> expected count = %d junc x %d span layers = %d"
      % (len(set(bb.web_ia) | set(bb.web_ib)), NS - 1, len(set(bb.web_ia) | set(bb.web_ib)) * (NS - 1)))
oth = [e for e in ecount if cls(e)[0] == "OTHER"]
print("  edges connecting NON-adjacent planes / bad topology = %d" % len(oth))

# every quad spans exactly two consecutive layers?
pl = quads // Ntot
badlayer = np.where(~((pl.max(1) - pl.min(1)) == 1))[0]
print("  quads NOT spanning exactly 2 consecutive layers = %d" % len(badlayer))
used = np.zeros(len(nodes), bool); used[quads.ravel()] = True
print("  unused (orphan) nodes = %d" % int((~used).sum()))
# is any layer isolated? -> check every plane index appears in some quad on both sides
lay_ok = all(((pl.min(1) == p).any() and (pl.max(1) == p + 1).any()) for p in range(NS - 1))
print("  every consecutive layer pair connected by quads: %s" % lay_ok)

# ---------------------------------------------------------------- (b) duplicate nodes
print("\n=== (b) COINCIDENT / UNMERGED NODES ===")
tr = cKDTree(nodes)
prs = tr.query_pairs(1e-9, output_type='ndarray')
print("  coincident node pairs within 1e-9 = %d" % len(prs))
for tol in (1e-6, 1e-4, 1e-3):
    print("    within %.0e = %d" % (tol, len(tr.query_pairs(tol, output_type='ndarray'))))
if len(prs):
    sn = np.unique(prs % Ntot); pn = np.unique(prs // Ntot)
    print("    section-node ids involved: %s" % sn[:40])
    print("    plane ids involved (min,max): %d %d" % (pn.min(), pn.max()))
    # are the duplicates connected (share a quad) or genuinely split?
    inc = defaultdict(set)
    for e, q in enumerate(quads):
        for n_ in q:
            inc[int(n_)].add(e)
    split = sum(1 for a, b in prs if not (inc[int(a)] & inc[int(b)]))
    print("    of those, pairs sharing NO quad (=> genuine split) = %d" % split)
# also check section-level duplicates at each real station (degenerate section polygon)
for i in (0, 5, 25, 45, 50):
    P = bl["Pk"][i]
    d = np.linalg.norm(P[bb.sec_elems[:, 0]] - P[bb.sec_elems[:, 1]], axis=1)
    print("    station %2d: section elem length min=%.3e max=%.3e ; zero-length elems=%d"
          % (i, d.min(), d.max(), int((d < 1e-9).sum())))

# ---------------------------------------------------------------- (c) root BC
print("\n=== (c) ROOT BC ===")
rn = np.unique(np.asarray(root) // 6)
print("  clamped dof = %d ; clamped nodes = %d ; node id range [%d,%d]"
      % (len(root), len(rn), rn.min(), rn.max()))
print("  clamped set == exactly layer 0 (nodes 0..%d): %s" % (Ntot - 1, np.array_equal(rn, np.arange(Ntot))))
print("  all 6 dof per clamped node: %s" % (len(root) == 6 * len(rn)))
print("  x of layer 0: min=%.4f max=%.4f ; global x min=%.4f max=%.4f"
      % (nodes[:Ntot, 0].min(), nodes[:Ntot, 0].max(), nodes[:, 0].min(), nodes[:, 0].max()))
print("  layer 0 IS the global min-x layer: %s" % bool(np.isclose(nodes[:Ntot, 0].max(), nodes[:, 0].min())))

# ---------------------------------------------------------------- helpers
def moment_from_N(Ne, p, i):
    P = bl["Pk"][i]; M = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zmid = 0.5 * (P[a, 1] + P[b, 1])
        M += -Ne[p * NSE + se, 0] * zmid * ds
    return M


def axial_from_N(Ne, p, i):
    P = bl["Pk"][i]; F = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        F += Ne[p * NSE + se, 0] * np.linalg.norm(P[b] - P[a])
    return F


def EI_station(i):
    """oint A11 z^2 ds about the A11-weighted centroid (from the SAME ABD the FE uses)."""
    P = bl["Pk"][i]; A = bl["Ak"][i]
    ds = np.linalg.norm(P[bb.sec_elems[:, 1]] - P[bb.sec_elems[:, 0]], axis=1)
    zm = 0.5 * (P[bb.sec_elems[:, 0], 1] + P[bb.sec_elems[:, 1], 1])
    a11 = A[:, 0, 0]
    ea = (a11 * ds).sum(); zc = (a11 * ds * zm).sum() / ea
    return (a11 * ds * (zm - zc) ** 2).sum(), ea


# ---------------------------------------------------------------- (f) element RBM patch test
print("\n=== (f) ELEMENT RIGID-BODY-MOTION PATCH TEST of _L_lg ===")
rng = np.random.default_rng(0)
Rr = np.linalg.qr(rng.normal(size=(3, 3)))[0]                # random rotation -> tilted element plane
q4 = np.array([[0, 0, 0], [1.3, 0, 0], [1.35, 0.9, 0], [0.05, 0.85, 0]])
Xe = q4 @ Rr.T + np.array([2.0, -1.0, 0.5])
T, xyl = sb._elem_frame(Xe, [0, 1, 2, 3])
ABD1, Gs1 = sb._iso_ABD(3e10, 0.3, 0.02)
Ke, _ = sb.element_K_KG(xyl, ABD1, Gs1, np.zeros(3))
L_code = sb._L_lg(T)
L_fix = L_code.copy()
for a in range(4):                                          # beta_x = e2.r , beta_y = -e1.r
    L_fix[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
    L_fix[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
scale = np.mean(np.abs(np.diag(Ke)))
print("  mode                    E_code/E_scale        E_fixed/E_scale")
for nm, tv, rv in [("trans x", [1, 0, 0], [0, 0, 0]), ("trans z", [0, 0, 1], [0, 0, 0]),
                   ("rot  x", [0, 0, 0], [1, 0, 0]), ("rot  y", [0, 0, 0], [0, 1, 0]),
                   ("rot  z(drill)", [0, 0, 0], [0, 0, 1]), ("rot rand", [0, 0, 0], list(rng.normal(size=3)))]:
    tv = np.array(tv, float); rv = np.array(rv, float)
    ug = np.zeros(24)
    for a in range(4):
        ug[6 * a:6 * a + 3] = tv + np.cross(rv, Xe[a]); ug[6 * a + 3:6 * a + 6] = rv
    nrm = (ug @ ug) * scale + 1e-30
    Ec = (L_code @ ug) @ Ke @ (L_code @ ug) / nrm
    Ef = (L_fix @ ug) @ Ke @ (L_fix @ ug) / nrm
    print("  %-14s %20.6e %20.6e" % (nm, Ec, Ef))

# ---------------------------------------------------------------- static solves
print("\n=== assembling K (%d dof) ===" % (6 * len(nodes)))
t1 = time.time(); K = sb.assemble_K(nodes, quads, ABD_e, Gs_e); print("  %.1fs" % (time.time() - t1))

f_tr = bb.traction_load(nodes, quads)
u_tr = sb.solve_static(nodes, quads, ABD_e, Gs_e, f_tr, root, K=K)
N_tr = sb.element_membrane_N(nodes, quads, ABD_e, u_tr)
FF = bb.beam_forces_from_traction(nodes, f_tr, bl["Rk"])
itip = int(np.argmax(nodes[:, 0]))
print("  traction: tip flap disp = %.5f m" % u_tr[6 * itip + 2])

# ---- (d) PURE TIP TRANSVERSE LOAD ----
print("\n=== (d) PURE TIP TRANSVERSE LOAD (decisive) ===")
FTIP = 1.0e6
tip0 = (NS - 1) * Ntot
f_tip = np.zeros(6 * len(nodes))
for n_ in range(tip0, tip0 + Ntot):
    f_tip[6 * n_ + 2] += FTIP / Ntot
u_tip = sb.solve_static(nodes, quads, ABD_e, Gs_e, f_tip, root, K=K)
N_tip = sb.element_membrane_N(nodes, quads, ABD_e, u_tip)
FFt = bb.beam_forces_from_traction(nodes, f_tip, bl["Rk"])
print("  tip disp = %.5f m ; applied Fz=%.4e" % (u_tip[6 * itip + 2], f_tip.reshape(-1, 6)[:, 2].sum()))
print("\n sta      x       FF_My(applied)    M_y(FE)        FE/FF      oint N11 ds    EI(oint A11 z^2)")
for i in [5, 15, 25, 35, 45]:
    p = min(i * MPER, NS - 2)
    m = moment_from_N(N_tip, p, i); ff = FFt[i][4]
    ei, ea = EI_station(i)
    print("  %2d  %7.2f   %+.4e   %+.4e   %8.4f   %+.3e   %.4e"
          % (i, bl["Rk"][i], ff, m, m / ff, axial_from_N(N_tip, p, i), ei))

# beam-theory tip deflection for the SAME tip load, from the FE's own A11
xs = bl["Rk"]; EIv = np.array([EI_station(i)[0] for i in range(NSTA)])
Ltip = xs[-1]
trap = getattr(np, "trapezoid", np.trapz)
integ = trap(FTIP * (Ltip - xs) * (Ltip - xs) / EIv, xs)
print("\n  beam-theory tip deflection (EI from the same A11) = %.5f m" % integ)
print("  FE tip deflection                                 = %.5f m   (FE/beam = %.4f)"
      % (u_tip[6 * itip + 2], u_tip[6 * itip + 2] / integ))
# traction-case beam deflection
Mx = np.array([FF[i][4] for i in range(NSTA)])
w_tr = trap(-Mx * (Ltip - xs) / EIv, xs)
print("  traction case: beam-theory tip deflection = %.5f m ; FE = %.5f m  (FE/beam = %.4f)"
      % (w_tr, u_tr[6 * itip + 2], u_tr[6 * itip + 2] / w_tr))

# ---------------------------------------------------------------- (e) energy split
print("\n=== (e) STRAIN-ENERGY SPLIT of the FE static solutions ===")
def esplit(u):
    Em = Eb = Es = 0.0
    for e, q in enumerate(quads):
        T_, xyl_ = sb._elem_frame(nodes, q); L = sb._L_lg(T_)
        ul = L @ np.concatenate([u[6 * n_:6 * n_ + 6] for n_ in q])
        A = ABD_e[e][:3, :3]; Bc = ABD_e[e][:3, 3:]; D = ABD_e[e][3:, 3:]; Gs = Gs_e[e]
        _, _, BsA, _, _, JA, _ = sb._B_at(xyl_, 0.0, -1.0); gxiA = (JA @ BsA)[0]
        _, _, BsC, _, _, JC, _ = sb._B_at(xyl_, 0.0, 1.0); gxiC = (JC @ BsC)[0]
        _, _, BsB, _, _, JB, _ = sb._B_at(xyl_, 1.0, 0.0); getB = (JB @ BsB)[1]
        _, _, BsD, _, _, JD, _ = sb._B_at(xyl_, -1.0, 0.0); getD = (JD @ BsD)[1]
        for xi in sb.G2:
            for eta in sb.G2:
                Bm, Bb_, _, _, detJ, Jg, _ = sb._B_at(xyl_, xi, eta)
                em = Bm @ ul; kb = Bb_ @ ul
                Em += em @ A @ em * detJ; Eb += kb @ D @ kb * detJ
                Bgxi = 0.5 * (1 - eta) * gxiA + 0.5 * (1 + eta) * gxiC
                Bget = 0.5 * (1 + xi) * getB + 0.5 * (1 - xi) * getD
                Bsc = np.linalg.inv(Jg) @ np.vstack([Bgxi, Bget])
                gs = Bsc @ ul
                Es += gs @ Gs @ gs * detJ
    tot = Em + Eb + Es
    return Em / tot, Eb / tot, Es / tot, tot
for nm, uu in [("traction", u_tr), ("tip load", u_tip)]:
    fm, fb, fs, tt = esplit(uu)
    print("  %-9s  membrane %6.2f%%   bending %6.2f%%   TRANSVERSE SHEAR %6.2f%%   (2U=%.4e)"
          % (nm, 100 * fm, 100 * fb, 100 * fs, tt))

print("\n=== traction case: FE section-equilibrium (same table as dbg_blade_equil) ===")
for i in [5, 15, 25, 35, 45]:
    p = min(i * MPER, NS - 2)
    m = moment_from_N(N_tr, p, i); ff = FF[i][4]
    print("  sta %2d  FF_My=%+.4e  M_FE=%+.4e  ratio=%8.4f" % (i, ff, m, m / ff))
print("\ndone %.1fs" % (time.time() - t0))
