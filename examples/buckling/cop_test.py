"""cop_test.py -- verify the COPLANAR-ONLY drilling spring fixes the square WITHOUT breaking the cylinder.
Expect: cylinder ~0.999*classical (all nodes flat -> penalized as before); prismatic square a=1 -> ~2.31e7
= analytical k=4 = FSM (corner nodes are folds -> no spring -> no leak)."""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

L, t = 2.0, 0.02; nc, nl, M = 160, 80, 16
E, nu = 200e9, 0.3
D = E * t**3 / (12 * (1 - nu**2))

# ---- cylinder ----
R = 1.0; Ncl = E * t**2 / (R * np.sqrt(3 * (1 - nu**2)))
ABDc, Gsc = sb._iso_ABD(E, nu, t)
th = np.linspace(0, 2 * np.pi, nc, endpoint=False); xs = np.linspace(0, L, nl + 1)
nod = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(nl + 1) for j in range(nc)])
ix = lambda i, j: i * nc + (j % nc)
qd = np.array([[ix(i, j), ix(i + 1, j), ix(i + 1, j + 1), ix(i, j + 1)] for i in range(nl) for j in range(nc)])
ne = len(qd); Ac = np.repeat(ABDc[None], ne, 0); Gc = np.repeat(Gsc[None], ne, 0)
Nc = np.repeat(np.array([-1.0, 0, 0])[None], ne, 0)
fx = []
for j in range(nc):
    r0, rL = ix(0, j), ix(nl, j); fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
lc = sb.solve_buckling(nod, qd, Ac, Gc, Nc, np.unique(fx), n_modes=6)[0]
fmask = sb._flat_node_mask(nod, qd)
print("CYLINDER nc=%d : N_cr=%.4e  ratio=%.4f   flat nodes %d/%d" % (nc, lc[0], lc[0] / Ncl, fmask.sum(), len(nod)))

# ---- prismatic square a=1 ----
a = 1.0; nps = nc // 4; cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
ring = np.array([np.array(cor[k], float) + (j / nps) * (np.array(cor[(k + 1) % 4], float) - np.array(cor[k], float))
                 for k in range(4) for j in range(nps)])
strips = np.array([[i, (i + 1) % (4 * nps)] for i in range(4 * nps)])
Napp = -1.0 / (4 * a)
lamF = np.asarray(fsm.solve_fsm_multi(ring, strips, [ABDc] * len(strips),
                                      [np.array([Napp, 0.0, 0.0])] * len(strips), L, M, n_modes=4))
nod2 = np.array([[xs[i], ring[p, 0], ring[p, 1]] for i in range(nl + 1) for p in range(nc)])
qd2 = np.array([[ix(i, p), ix(i + 1, p), ix(i + 1, p + 1), ix(i, p + 1)] for i in range(nl) for p in range(nc)])
A2 = np.repeat(ABDc[None], ne, 0); G2 = np.repeat(Gsc[None], ne, 0); N2 = np.repeat(np.array([Napp, 0, 0])[None], ne, 0)
fx2 = []
for p in range(nc):
    r0, rL = ix(0, p), ix(nl, p); fx2 += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
ls = sb.solve_buckling(nod2, qd2, A2, G2, N2, np.unique(fx2), n_modes=6)[0]
Pcl = 16 * np.pi**2 * D / a
fmask2 = sb._flat_node_mask(nod2, qd2)
print("SQUARE a=1 : FEA=%.4e (k=4 analytic=%.4e, FEA/an=%.3f)  FSM=%.4e  FSM/FEA=%.3f   flat %d/%d"
      % (ls[0], Pcl, ls[0] / Pcl, lamF[0], lamF[0] / ls[0], fmask2.sum(), len(nod2)))
