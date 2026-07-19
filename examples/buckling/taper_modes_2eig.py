"""taper_modes_2eig.py -- FIRST TWO buckling eigenvalues of tapered thin-walled tubes represented by
MULTIPLE cross-sections, JAX-Shell FEA (full 3-D) vs OpenSG-RM multi-harmonic FSM (3 cross-sections).

Two geometries, each iso + [+-45]s wall, t=0.02, L=2, size s: 1.0 (root) -> 0.5 (tip):
  CONE   : circular ring, size = radius R;   perimeter 2*pi*R ; meridional cos = L/sqrt(L^2+(R1-R2)^2)
  SQUARE : square  ring, size = side  a;     perimeter 4*a     ; meridional cos = L/sqrt(L^2+((a1-a2)/2)^2)

Pre-buckling: uniform axial P, meridional membrane N(x) = -1/(perimeter(s(x)) * cos) imposed on BOTH
pathways (isolates the buckling formulation).  The FSM buckles each of 3 cross-sections (root/mid/tip)
independently; the taper's eigenvalues are the two lowest over the pooled section spectra.  Reports which
section governs -- expected: cone ~R-independent (tip marginal), square ROOT governs (P_cr ~ 1/a)."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

L, t = 2.0, 0.02
nc, nl, M = 160, 80, 16
ABD_iso = fsm.iso_abd(200e9, 0.3, t); Gs_iso = (5. / 6.) * (200e9 / (2 * 1.3)) * t * np.eye(2)
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ABD_m45 = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)
Gs_m45 = (5. / 6.) * 5e9 * t * np.eye(2)


def sq_ring(a, n):
    """square perimeter, side a, n nodes (n/4 per side, starting corner included -> no quad straddles a corner)."""
    nps = n // 4
    cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
    pts = []
    for k in range(4):
        P0 = np.array(cor[k], float); P1 = np.array(cor[(k + 1) % 4], float)
        for j in range(nps):
            pts.append(P0 + (j / nps) * (P1 - P0))
    pts = np.array(pts); strips = np.array([[i, (i + 1) % len(pts)] for i in range(len(pts))])
    return pts, strips


GEOM = {
    "cone":   dict(ring=lambda s, n: fsm.cyl_ring(s, n), perim=lambda s: 2 * np.pi * s,
                   cos=L / np.sqrt(L**2 + (1.0 - 0.5)**2)),
    "square": dict(ring=sq_ring, perim=lambda s: 4 * s,
                   cos=L / np.sqrt(L**2 + ((1.0 - 0.5) / 2)**2)),
}
S1, S2 = 1.0, 0.5
Sx = lambda x: S1 + (S2 - S1) * x / L


def fea_two(g, ABD, Gs):
    Nmer = lambda s: -1.0 / (g["perim"](s) * g["cos"])
    xs = np.linspace(0, L, nl + 1)
    rings = [g["ring"](Sx(xs[i]), nc)[0] for i in range(nl + 1)]
    nodes = np.array([[xs[i], rings[i][p, 0], rings[i][p, 1]] for i in range(nl + 1) for p in range(nc)])
    idx = lambda i, p: i * nc + (p % nc)
    quads = np.array([[idx(i, p), idx(i + 1, p), idx(i + 1, p + 1), idx(i, p + 1)]
                      for i in range(nl) for p in range(nc)])
    ne = len(quads); ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec = np.array([[Nmer(Sx(0.5 * (xs[e // nc] + xs[e // nc + 1]))), 0.0, 0.0] for e in range(ne)])
    fx = []
    for p in range(nc):
        r0, rL = idx(0, p), idx(nl, p)
        fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]       # SS3 both ends
    loads = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, np.unique(fx), n_modes=6)[0]
    return loads[0], loads[1]


def fsm_sections(g, ABD):
    Nmer = lambda s: -1.0 / (g["perim"](s) * g["cos"])
    out = []
    for x in [0.0, L / 2, L]:
        s = Sx(x); ring, strips = g["ring"](s, nc)
        N_s = [np.array([Nmer(s), 0.0, 0.0])] * len(strips)
        lam = fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, M, n_modes=4)
        out.append((x, s, np.asarray(lam)))
    return out


print("tapered tubes  size %.2f->%.2f  L=%.1f t=%.3f" % (S1, S2, L, t))
for gname, g in GEOM.items():
    print("\n===== %s  (cos=%.3f) =====" % (gname.upper(), g["cos"]))
    for tag, ABD, Gs in [("iso", ABD_iso, Gs_iso), ("m45", ABD_m45, Gs_m45)]:
        t0 = time.time(); f1, f2 = fea_two(g, ABD, Gs); tf = time.time() - t0
        sec = fsm_sections(g, ABD)
        pool = sorted((l, x, s) for x, s, lam in sec for l in lam)     # (factor, x, size)
        p1, p2 = pool[0], pool[1]
        gx = {0.0: "root", 1.0: "mid", 2.0: "tip"}.get(p1[1], "x=%.1f" % p1[1])
        print(" %s: FEA [%.4e, %.4e] | FSM [%.4e, %.4e] | FSM/FEA lam1=%.3f  (gov=%s, FEA %.0fs)"
              % (tag, f1, f2, p1[0], p2[0], p1[0] / f1, gx, tf))
        for x, s, lam in sec:
            print("     sec x=%.1f size=%.3f : FSM lam1,2 = %.4e %.4e" % (x, s, lam[0], lam[1]))
