"""sq_check.py -- diagnostic: is the square-tube FSM<->FEA discrepancy a BUG or the per-station taper limit?
Run PRISMATIC square tubes (constant side a) at a=1.0 and a=0.5.  If prismatic FEA ~= FSM (and ~= the
analytical SS-plate value 4 pi^2 D/b^2 * 4a), then the square mesh/FSM are correct and the TAPERED
discrepancy is purely the per-station approximation (buckle half-wavelength ~ wall width ~ taper length).
Also locate where the TAPERED FEA buckle sits along the span."""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

L, t = 2.0, 0.02; nc, nl, M = 160, 80, 16
E, nu = 200e9, 0.3
ABD = fsm.iso_abd(E, nu, t); Gs = (5. / 6.) * (E / (2 * 1.3)) * t * np.eye(2)
D = E * t**3 / (12 * (1 - nu**2))


def sq_ring(a, n):
    nps = n // 4; cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
    pts = []
    for k in range(4):
        P0 = np.array(cor[k], float); P1 = np.array(cor[(k + 1) % 4], float)
        for j in range(nps):
            pts.append(P0 + (j / nps) * (P1 - P0))
    pts = np.array(pts); strips = np.array([[i, (i + 1) % len(pts)] for i in range(len(pts))])
    return pts, strips


def prismatic(a):
    Napp = -1.0 / (4 * a)                                    # uniform axial membrane per unit P (cos=1)
    ring, strips = sq_ring(a, nc)
    N_s = [np.array([Napp, 0.0, 0.0])] * len(strips)
    lamF = np.asarray(fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, M, n_modes=4))
    xs = np.linspace(0, L, nl + 1)
    nodes = np.array([[xs[i], ring[p, 0], ring[p, 1]] for i in range(nl + 1) for p in range(nc)])
    idx = lambda i, p: i * nc + (p % nc)
    quads = np.array([[idx(i, p), idx(i + 1, p), idx(i + 1, p + 1), idx(i, p + 1)]
                      for i in range(nl) for p in range(nc)])
    ne = len(quads); ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec = np.repeat(np.array([Napp, 0.0, 0.0])[None], ne, 0)
    fx = []
    for p in range(nc):
        r0, rL = idx(0, p), idx(nl, p); fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
    loads, modes = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, np.unique(fx), n_modes=6)
    # analytical isolated SS wall: N_cr = 4 pi^2 D / a^2 ; P_cr = N_cr / |Napp| = 16 pi^2 D / a
    Pcl = 16 * np.pi**2 * D / a
    return lamF, loads, Pcl


for a in [1.0, 0.5]:
    lamF, loads, Pcl = prismatic(a)
    print("PRISMATIC square a=%.2f : FEA[%.4e,%.4e] FSM[%.4e,%.4e] analytic(4pi2D/b2)=%.4e | FEA/an=%.3f FSM/an=%.3f FSM/FEA=%.3f"
          % (a, loads[0], loads[1], lamF[0], lamF[1], Pcl, loads[0] / Pcl, lamF[0] / Pcl, lamF[0] / loads[0]))
