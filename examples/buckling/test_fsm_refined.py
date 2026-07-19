"""test_fsm_refined.py -- refine the mesh to separate the FSM residual (transverse shear) from the
coarse-FEA facet error.  FSM (multi-harmonic, Kirchhoff) vs 3-D shell FEA at nc=160 and 240, iso + m45."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

R, t, L = 1.0, 0.02, 2.0
ABD_iso = fsm.iso_abd(200e9, 0.3, t); Gs_iso = (5. / 6.) * (200e9 / (2 * 1.3)) * t * np.eye(2)
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ABD_m45 = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)
Gs_m45 = (5. / 6.) * 5e9 * t * np.eye(2)
Ncl = 200e9 * t**2 / (R * np.sqrt(3 * (1 - 0.3**2)))


def cyl_fea(ABD, Gs, nc, nl):
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False); xs = np.linspace(0, L, nl + 1)
    nodes = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(nl + 1) for j in range(nc)])
    idx = lambda i, j: i * nc + (j % nc)
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(nl) for j in range(nc)])
    ne = len(quads)
    ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec = np.repeat(np.array([-1.0, 0, 0])[None], ne, 0)
    fx = []
    for j in range(nc):
        r0, rL = idx(0, j), idx(nl, j)
        fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
    return sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, np.unique(fx), n_modes=4)[0][0]


for tag, ABD, Gs in [("iso", ABD_iso, Gs_iso), ("m45", ABD_m45, Gs_m45)]:
    print("\n%s cylinder:" % tag)
    for nc in [160, 240]:
        ring, strips = fsm.cyl_ring(R, nc); N_s = [np.array([-1.0, 0, 0])] * len(strips)
        nf = fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, 16)[0]
        t0 = time.time(); ne = cyl_fea(ABD, Gs, nc, nc // 2); tf = time.time() - t0
        print("  nc=%3d : FSM(Kirchhoff)=%.4e  3D-FEA(RM)=%.4e  FSM/FEA=%.3f  (FEA %.0fs)"
              % (nc, nf, ne, nf / ne, tf))
