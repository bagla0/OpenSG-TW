"""cone_buckling.py -- tapered shell CONE axial local buckling.
  JAX-Shell FEA  : full 3-D shell cone, MITC4 RM element, imposed meridional pre-buckling N -> eigenvalue.
  OpenSG-RM-FSM  : 3 cross-sections (root R1, mid, tip R2); multi-harmonic FSM per section; cone factor =
                   MIN over the 3 sections.  Tests whether per-section FSM reproduces the full tapered FEA
                   (the connected/different-cross-section question).  iso + m45 [+-45]s walls.

Pre-buckling: uniform axial force P; meridional membrane N(x) = -P / (2 pi R(x) cos(alpha)).  We impose the
same analytic N on BOTH pathways, so the comparison isolates the buckling formulation (full 3-D vs per-
section FSM), not the stress recovery.  Reported N_cr is the axial-force factor at buckling (P_ref=1)."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

R1, R2, L, t = 1.0, 0.5, 2.0, 0.02
nc, nl, M = 160, 80, 16
cosa = L / np.sqrt(L**2 + (R1 - R2)**2)                       # meridian cos(half-angle)
ABD_iso = fsm.iso_abd(200e9, 0.3, t); Gs_iso = (5. / 6.) * (200e9 / (2 * 1.3)) * t * np.eye(2)
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ABD_m45 = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)
Gs_m45 = (5. / 6.) * 5e9 * t * np.eye(2)
Rx = lambda x: R1 + (R2 - R1) * x / L
Nmer = lambda R: -1.0 / (2 * np.pi * R * cosa)               # meridional N per unit axial P


def cone_fea(ABD, Gs):
    xs = np.linspace(0, L, nl + 1); th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    nodes = np.array([[xs[i], Rx(xs[i]) * np.cos(th[j]), Rx(xs[i]) * np.sin(th[j])]
                      for i in range(nl + 1) for j in range(nc)])
    idx = lambda i, j: i * nc + (j % nc)
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(nl) for j in range(nc)])
    ne = len(quads); ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec = np.array([[Nmer(Rx(0.5 * (xs[e // nc] + xs[e // nc + 1]))), 0.0, 0.0] for e in range(ne)])
    fx = []
    for j in range(nc):
        r0, rL = idx(0, j), idx(nl, j)
        fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]     # SS3 both ends
    return sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, np.unique(fx), n_modes=6)[0][0]


def cone_fsm(ABD):
    out = []
    for x in [0.0, L / 2, L]:
        R = Rx(x); ring, strips = fsm.cyl_ring(R, nc)
        N_s = [np.array([Nmer(R), 0.0, 0.0])] * len(strips)
        lam = fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, M)[0]
        out.append((x, R, lam))
    return out


print("tapered cone R1=%.2f R2=%.2f L=%.1f t=%.3f  cos(alpha)=%.3f" % (R1, R2, L, t, cosa))
for tag, ABD, Gs in [("iso", ABD_iso, Gs_iso), ("m45", ABD_m45, Gs_m45)]:
    t0 = time.time(); nfea = cone_fea(ABD, Gs); tf = time.time() - t0
    sec = cone_fsm(ABD); pmin = min(l for _, _, l in sec)
    print("\n%s cone: JAX-Shell FEA=%.4e | OpenSG-RM-FSM(3-sec min)=%.4e | FSM/FEA=%.3f  (FEA %.0fs)"
          % (tag, nfea, pmin, pmin / nfea, tf))
    for x, R, l in sec:
        print("   section x=%.1f R=%.3f : FSM N_cr=%.4e" % (x, R, l))
