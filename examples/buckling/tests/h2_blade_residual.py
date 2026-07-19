"""h2_blade_residual.py -- with _L_lg FIXED, WHERE does the rest of the blade section moment sit?

Exact FE free-body decomposition at a span cut p:
  assemble g = sum_{elements OUTBOARD of p} K_e u_e .  For nodes strictly outboard g = fext, so the
  interface force the inboard part applies on the outboard free body is  fcut = g - fext  on the cut ring.
  Then    M_section = sum_ring [ r x fcut[:3] ]  +  sum_ring fcut[3:6]
                       \_ carried by in-plane FORCES _/    \_ carried by wall BENDING couples _/
  and M_section must equal FF[i] identically (assembly identity) -> the split is the diagnostic.
Also: element aspect ratios, and a span-refinement (MPER) sweep of the membrane-moment ratio.
"""
import os, sys, time
import numpy as np
import scipy.sparse as spx
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


sb._L_lg = _L_lg_fixed
import blade_iso as bi
import blade_buckling as bb
NSE = bi.NSE


def run(MPER):
    bb.MPER = MPER; bi.MPER = MPER
    bl = bi.build()
    nodes, quads = bl["nodes"], bl["quads"]
    ABD_e, Gs_e, root = bl["ABD_e"], bl["Gs_e"], bl["root"]
    f = bb.traction_load(nodes, quads)
    FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
    u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root)
    Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
    return bl, nodes, quads, ABD_e, Gs_e, f, FF, u, Nf


print("=" * 100)
print("element ASPECT RATIO along the span (span-length / arc-length), MPER=2")
print("=" * 100)
bl, nodes, quads, ABD_e, Gs_e, f, FF, u, Nf = run(2)
Ntot = bb.Ntot; NS = bl["NS"]
print("  mesh %d nodes %d quads   tip flap = %.3f m" % (len(nodes), len(quads), u[6 * int(np.argmax(nodes[:, 0])) + 2]))
print("  %6s %9s | %10s %10s %10s" % ("plane", "x[m]", "span_len", "med arc", "aspect"))
for p in [2, 10, 30, 50, 70, 90, 98]:
    e0 = p * NSE
    sl = np.linalg.norm(nodes[quads[e0][1]] - nodes[quads[e0][0]])
    arcs = [np.linalg.norm(nodes[quads[e0 + se][3]] - nodes[quads[e0 + se][0]]) for se in range(NSE)]
    print("  %6d %9.2f | %10.4f %10.4f %10.1f" % (p, nodes[quads[e0][0], 0], sl, np.median(arcs), sl / np.median(arcs)))

print()
print("=" * 100)
print("EXACT free-body split of the section moment (MPER=2, _L_lg FIXED)")
print("=" * 100)
K_free = None
ne = len(quads)
# per-element K_e u_e -> global g, restricted to outboard element sets
Kloc = np.zeros((ne, 24, 24)); GD = np.zeros((ne, 24), int)
flat = sb._flat_node_mask(nodes, quads)
for e, q in enumerate(quads):
    T, xyl = sb._elem_frame(nodes, q)
    Ke, _ = sb.element_K_KG(xyl, ABD_e[e], Gs_e[e], np.zeros(3))
    L = sb._L_lg(T); Kd = L.T @ Ke @ L
    kdr = sb._KDR_SCALE * np.mean(np.abs(np.diag(Ke)[2::5]))
    Kn = kdr * np.outer(T[2], T[2])
    for a in range(4):
        if flat[q[a]]:
            Kd[6 * a + 3:6 * a + 6, 6 * a + 3:6 * a + 6] += Kn
    Kloc[e] = Kd; GD[e] = np.concatenate([np.arange(6 * n, 6 * n + 6) for n in q])
uel = u[GD]                                       # (ne,24)
fel = np.einsum("eij,ej->ei", Kloc, uel)          # (ne,24) element internal nodal forces

print("  %4s %10s | %14s %14s %14s | %8s %8s %8s" %
      ("sta", "FF_My", "M(forces)", "M(couples)", "M(total)", "F/FF", "C/FF", "tot/FF"))
for i in [5, 15, 25, 35, 45]:
    p = min(i * bb.MPER, NS - 2)
    g = np.zeros(6 * len(nodes))
    sel = np.arange(p * NSE, ne)                  # elements outboard of plane p
    np.add.at(g, GD[sel].ravel(), fel[sel].ravel())
    ring = np.arange(p * Ntot, (p + 1) * Ntot)
    dofs = (6 * ring[:, None] + np.arange(6)[None, :])
    fcut = g[dofs] - f[dofs]                       # (Ntot,6)
    r = nodes[ring] - np.array([bl["Rk"][i], 0.0, 0.0])
    Mf = np.cross(r, fcut[:, :3]).sum(0)
    Mc = fcut[:, 3:6].sum(0)
    ff = FF[i][4]
    print("  %4d %+.3e | %+14.5e %+14.5e %+14.5e | %8.4f %8.4f %8.4f"
          % (i, ff, Mf[1], Mc[1], Mf[1] + Mc[1], Mf[1] / ff, Mc[1] / ff, (Mf[1] + Mc[1]) / ff))
print("  (tot/FF must be ~ -1 by the assembly identity; the FF convention is opposite-signed.)")

print()
print("=" * 100)
print("SPAN-REFINEMENT sweep of the membrane-moment ratio  M_y(-oint N11 z ds)/FF_My")
print("=" * 100)


def mom(bl_, Nf_, p, i):
    P = bl_["Pk"][i]; M = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zm = 0.5 * (P[a, 1] + P[b, 1])
        M += -Nf_[p * NSE + se, 0] * zm * ds
    return M


for mp in [1, 2, 4, 8]:
    t0 = time.time()
    bl2, nodes2, quads2, A2, G2_, f2, FF2, u2, Nf2 = run(mp)
    row = []
    for i in [5, 15, 25, 35, 45]:
        p = min(i * mp, bl2["NS"] - 2)
        row.append(mom(bl2, Nf2, p, i) / FF2[i][4])
    print("  MPER=%d  (%d nodes, tip=%6.3f m, %3.0fs):  ratios @ sta 5/15/25/35/45 = %s"
          % (mp, len(nodes2), u2[6 * int(np.argmax(nodes2[:, 0])) + 2], time.time() - t0,
             "  ".join("%7.4f" % v for v in row)))
