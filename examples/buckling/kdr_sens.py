"""kdr_sens.py -- is the SS3 cylinder buckling load INDEPENDENT of the drilling-penalty magnitude?
A correct drilling stabilization removes the flat/soft drilling mode without polluting the eigenvalue,
so N_cr should be flat as _KDR_SCALE -> 0 (until it is too small to desingularize).  Uses the NEW
about-the-normal penalty."""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import shell_buckling as sb

R, t, L = 1.0, 0.02, 2.0; nc, nl = 160, 80
E, nu = 200e9, 0.3; Ncl = E * t**2 / (R * np.sqrt(3 * (1 - nu**2)))
ABD, Gs = sb._iso_ABD(E, nu, t)
th = np.linspace(0, 2 * np.pi, nc, endpoint=False); xs = np.linspace(0, L, nl + 1)
nodes = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(nl + 1) for j in range(nc)])
idx = lambda i, j: i * nc + (j % nc)
quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(nl) for j in range(nc)])
ne = len(quads); ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
Nvec = np.repeat(np.array([-1.0, 0, 0])[None], ne, 0)
fx = []
for j in range(nc):
    r0, rL = idx(0, j), idx(nl, j); fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
fx = np.unique(fx)
print("cylinder SS3  nc=%d nl=%d   classical N_cr=%.4e" % (nc, nl, Ncl))
for scale in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
    sb._KDR_SCALE = scale
    try:
        loads = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, fx, n_modes=6)[0]
        print("  kdr=%.0e  N_cr=%.5e  ratio=%.4f" % (scale, loads[0], loads[0] / Ncl))
    except Exception as ex:
        print("  kdr=%.0e  FAILED (%s)" % (scale, type(ex).__name__))
