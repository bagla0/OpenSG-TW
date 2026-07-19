"""sq_kdr.py -- does the PRISMATIC square-tube FEA buckling drop toward the analytical k=4 (2.31e7 = FSM)
as the drilling penalty k_dr -> 0?  At a 90-deg corner one wall's drilling axis = the adjacent wall's
physical bending-rotation axis, so a finite k_dr leaks into a real DOF and clamps the corner (k~6.8).
A smooth cylinder has no corners -> was k_dr-insensitive.  Expect: square FEA 3.95e7 (k_dr=1e-3) -> ~2.31e7."""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

L, t = 2.0, 0.02; nc, nl, M = 160, 80, 16
E, nu = 200e9, 0.3
ABD = fsm.iso_abd(E, nu, t); Gs = (5. / 6.) * (E / (2 * 1.3)) * t * np.eye(2)
D = E * t**3 / (12 * (1 - nu**2)); a = 1.0
Pcl = 16 * np.pi**2 * D / a                              # isolated SS wall (k=4) -> P_cr


def sq_ring(a, n):
    nps = n // 4; cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
    pts = []
    for k in range(4):
        P0 = np.array(cor[k], float); P1 = np.array(cor[(k + 1) % 4], float)
        for j in range(nps):
            pts.append(P0 + (j / nps) * (P1 - P0))
    return np.array(pts), np.array([[i, (i + 1) % (4 * nps)] for i in range(4 * nps)])


ring, strips = sq_ring(a, nc)
Napp = -1.0 / (4 * a)
lamF = np.asarray(fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips),
                                      [np.array([Napp, 0.0, 0.0])] * len(strips), L, M, n_modes=4))
xs = np.linspace(0, L, nl + 1)
nodes = np.array([[xs[i], ring[p, 0], ring[p, 1]] for i in range(nl + 1) for p in range(nc)])
idx = lambda i, p: i * nc + (p % nc)
quads = np.array([[idx(i, p), idx(i + 1, p), idx(i + 1, p + 1), idx(i, p + 1)] for i in range(nl) for p in range(nc)])
ne = len(quads); ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
Nvec = np.repeat(np.array([Napp, 0.0, 0.0])[None], ne, 0)
fx = []
for p in range(nc):
    r0, rL = idx(0, p), idx(nl, p); fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
fx = np.unique(fx)
print("prismatic square a=%.2f  analytic k=4 P_cr=%.4e   FSM=%.4e (FSM/an=%.3f)" % (a, Pcl, lamF[0], lamF[0] / Pcl))
for scale in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    sb._KDR_SCALE = scale
    try:
        loads = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, fx, n_modes=6)[0]
        print("  kdr=%.0e  FEA=%.4e  FEA/an=%.3f  FSM/FEA=%.3f" % (scale, loads[0], loads[0] / Pcl, lamF[0] / loads[0]))
    except Exception as ex:
        print("  kdr=%.0e  FAILED (%s)" % (scale, type(ex).__name__))
