"""test_fsm_multi.py -- multi-harmonic FSM closes the anisotropic bend-twist gap.
Sweep the number of harmonics M for iso and m45 [+-45]s cylinder (length L=2), compare to 3-D shell FEA.
Single harmonic (orthotropic core) over-predicts m45 ~20%; adding coupled harmonics should converge to FEA."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

R, t, L, nc, nl = 1.0, 0.02, 2.0, 160, 80
ring, strips = fsm.cyl_ring(R, nc)
N_s = [np.array([-1.0, 0.0, 0.0])] * len(strips)
ABD_iso = fsm.iso_abd(200e9, 0.3, t); Gs_iso = (5. / 6.) * (200e9 / 2.6) * t * np.eye(2)
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ABD_m45 = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)
Gs_m45 = (5. / 6.) * 5e9 * t * np.eye(2)


def fea_cyl(ABD, Gs):                              # 3-D SS3 shell FEA reference (corrected coplanar drilling)
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False); xs = np.linspace(0, L, nl + 1)
    nod = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(nl + 1) for j in range(nc)])
    ix = lambda i, j: i * nc + (j % nc)
    qd = np.array([[ix(i, j), ix(i + 1, j), ix(i + 1, j + 1), ix(i, j + 1)] for i in range(nl) for j in range(nc)])
    ne = len(qd); Ae = np.repeat(ABD[None], ne, 0); Ge = np.repeat(Gs[None], ne, 0)
    Nv = np.repeat(np.array([-1.0, 0, 0])[None], ne, 0); fx = []
    for j in range(nc):
        r0, rL = ix(0, j), ix(nl, j); fx += [6 * r0 + 1, 6 * r0 + 2, 6 * r0, 6 * rL + 1, 6 * rL + 2]
    return sb.solve_buckling(nod, qd, Ae, Ge, Nv, np.unique(fx), n_modes=6)[0][0]


FEA = {"iso": fea_cyl(ABD_iso, Gs_iso), "m45": fea_cyl(ABD_m45, Gs_m45)}

for tag, ABD in [("iso", ABD_iso), ("m45", ABD_m45)]:
    print("\n%s cylinder (3-D FEA N_cr = %.4e):" % (tag, FEA[tag]))
    for M in [1, 4, 8, 12, 16, 20]:
        t0 = time.time()
        lam = fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, M)
        print("  M=%2d : N_cr=%.4e  FSM/FEA=%.3f  (%.1fs)" % (M, lam[0], lam[0] / FEA[tag], time.time() - t0))
