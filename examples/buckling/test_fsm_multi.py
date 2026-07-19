"""test_fsm_multi.py -- multi-harmonic FSM closes the anisotropic bend-twist gap.
Sweep the number of harmonics M for iso and m45 [+-45]s cylinder (length L=2), compare to 3-D shell FEA.
Single harmonic (orthotropic core) over-predicts m45 ~20%; adding coupled harmonics should converge to FEA."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm

R, t, L, nc = 1.0, 0.02, 2.0, 160
ring, strips = fsm.cyl_ring(R, nc)
N_s = [np.array([-1.0, 0.0, 0.0])] * len(strips)
ABD_iso = fsm.iso_abd(200e9, 0.3, t)
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ABD_m45 = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)
FEA = {"iso": 4.6115e7, "m45": 5.4734e6}          # 3-D shell FEA (SS3 cylinder), from test_fsm_cyl.py

for tag, ABD in [("iso", ABD_iso), ("m45", ABD_m45)]:
    print("\n%s cylinder (3-D FEA N_cr = %.4e):" % (tag, FEA[tag]))
    for M in [1, 4, 8, 12, 16, 20]:
        t0 = time.time()
        lam = fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, M)
        print("  M=%2d : N_cr=%.4e  FSM/FEA=%.3f  (%.1fs)" % (M, lam[0], lam[0] / FEA[tag], time.time() - t0))
