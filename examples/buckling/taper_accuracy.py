"""taper_accuracy.py -- DECISIVE test of why the per-station FSM under-predicts a TAPERED tube.

Two competing explanations for the tapered square's FSM/FEA = 0.82:
  (A) REGIME  -- the plate buckle half-wavelength is comparable to the taper length, so the wall narrows
                 within one buckle and the prismatic per-station estimate is a genuine lower bound.
  (B) DEFICIENCY -- something in the FSM/assembly is simply wrong for tapered geometry.

They are distinguishable:
  * Sweep the TAPER RATE s2/s1: 1.0 (prismatic control) -> 0.9 -> 0.75 -> 0.5.
    (A) predicts FSM/FEA -> 1.0 as the taper vanishes.  (B) predicts a persistent offset even at s2/s1=1.
  * Sweep the NUMBER OF CROSS-SECTIONS 3 -> 5 -> 9 at fixed taper.
    (A) predicts the per-station minimum keeps DROPPING (finer sampling finds a worse station) and does NOT
        converge to the FEA -- i.e. more stations make it MORE conservative, not more accurate.
    (B) would show erratic / non-monotone behaviour.
Also reports, per case, the critical half-wavelength vs the taper length (the regime parameter).
"""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb
import robust_checks as rc

L, t = 2.0, 0.02
nc, nl, M = 160, 80, 16
E, nu = 200e9, 0.3
ABD_iso = fsm.iso_abd(E, nu, t); Gs_iso = (5. / 6.) * (E / (2 * 1.3)) * t * np.eye(2)
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ABD_m45 = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)
Gs_m45 = (5. / 6.) * 5e9 * t * np.eye(2)


def sq_ring(a, n):
    nps = n // 4; cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
    pts = []
    for k in range(4):
        P0 = np.array(cor[k], float); P1 = np.array(cor[(k + 1) % 4], float)
        for j in range(nps):
            pts.append(P0 + (j / nps) * (P1 - P0))
    return np.array(pts), np.array([[i, (i + 1) % (4 * nps)] for i in range(4 * nps)])


GEOM = {
    "cone":   dict(ring=lambda s, n: fsm.cyl_ring(s, n), perim=lambda s: 2 * np.pi * s, half=lambda s: s),
    "square": dict(ring=sq_ring, perim=lambda s: 4 * s, half=lambda s: s / 2),
}
S1 = 1.0


def cosang(g, S2):
    return L / np.sqrt(L**2 + (g["half"](S1) - g["half"](S2))**2)


def fea(g, S2, ABD, Gs):
    ca = cosang(g, S2); Sx = lambda x: S1 + (S2 - S1) * x / L
    Nmer = lambda s: -1.0 / (g["perim"](s) * ca)
    xs = np.linspace(0, L, nl + 1)
    rings = [g["ring"](Sx(xs[i]), nc)[0] for i in range(nl + 1)]
    nodes = np.array([[xs[i], rings[i][p, 0], rings[i][p, 1]] for i in range(nl + 1) for p in range(nc)])
    ix = lambda i, p: i * nc + (p % nc)
    quads = np.array([[ix(i, p), ix(i + 1, p), ix(i + 1, p + 1), ix(i, p + 1)]
                      for i in range(nl) for p in range(nc)])
    ne = len(quads); Ae = np.repeat(ABD[None], ne, 0); Ge = np.repeat(Gs[None], ne, 0)
    Nv = np.array([[Nmer(Sx(0.5 * (xs[e // nc] + xs[e // nc + 1]))), 0.0, 0.0] for e in range(ne)])
    fx = []
    for p in range(nc):
        r0, rL = ix(0, p), ix(nl, p); fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
    return sb.solve_buckling(nodes, quads, Ae, Ge, Nv, np.unique(fx), n_modes=6)[0][0]


def fsm_min(g, S2, ABD, nsec):
    ca = cosang(g, S2); Sx = lambda x: S1 + (S2 - S1) * x / L
    Nmer = lambda s: -1.0 / (g["perim"](s) * ca)
    best = np.inf
    for x in np.linspace(0.0, L, nsec):
        s = Sx(x); ring, strips = g["ring"](s, nc)
        lam = fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips),
                                  [np.array([Nmer(s), 0.0, 0.0])] * len(strips), L, M)
        best = min(best, float(np.asarray(lam)[0]))
    return best


def a_crit(g, S2, ABD):
    """critical half-wavelength from a single-harmonic signature sweep at the governing (root/tip) station."""
    s = S1 if g is GEOM["square"] else S2                      # square: root governs; cone: tip
    ring, strips = g["ring"](s, nc); ca = cosang(g, S2)
    N_s = [np.array([-1.0 / (g["perim"](s) * ca), 0.0, 0.0])] * len(strips)
    aa = np.geomspace(0.06, 2.0, 26)
    _, lam1, ac, _ = fsm.signature_curve(ring, strips, [ABD] * len(strips), N_s, aa)
    return float(ac)


print("taper-rate + station-refinement study   (L=%.1f, t=%.3f, nc=%d)" % (L, t, nc))
for gname, g in GEOM.items():
    for tag, ABD, Gs in [("iso", ABD_iso, Gs_iso), ("m45", ABD_m45, Gs_m45)]:
        print("\n=== %s / %s ===" % (gname.upper(), tag))
        ac = a_crit(g, 0.5, ABD)
        print("  critical half-wavelength a* = %.3f m   (taper length L = %.1f m -> a*/L = %.3f)"
              % (ac, L, ac / L))
        print("   s2/s1 |    FEA      | FSM(3sec)  ratio | FSM(5sec)  ratio | FSM(9sec)  ratio")
        for S2 in [1.0, 0.9, 0.75, 0.5]:
            t0 = time.time(); F = fea(g, S2, ABD, Gs); dt = time.time() - t0
            row = "   %.2f  | %.4e |" % (S2, F)
            for nsec in (3, 5, 9):
                m = fsm_min(g, S2, ABD, nsec)
                row += " %.4e %.3f |" % (m, m / F)
            print(row + "  (FEA %.0fs)" % dt)
        reg = rc.fsm_regime(a_crit=ac, station_spacing=L / 2, width_at_station=g["half"](S1) * 2)
        print("  regime guard (3 sections -> spacing L/2): %s  a/spacing=%.2f" % (reg["verdict"], reg["a_over_spacing"]))
