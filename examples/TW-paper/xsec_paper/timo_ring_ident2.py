"""timo_ring_ident2.py -- confirm the closed-form set for the center-ref circular tube:
  Atilde_ij = A_ij - A_i2 A_j2 / A22   (hoop-relaxed driven block, i,j in {1,6})
  EA  = 2 pi R Atilde11        C14 = -/+ 2 pi R^2 Atilde16      GJ = 2 pi R^3 Atilde66*2?? (check)
  GA2 = GA3 = pi R Atilde66 (1 + h^2/(6R^2))
  EI2 = EI3 = pi R^3 Atilde11 + pi R Dtilde11   (Dtilde = D-block hoop-relaxed)
"""
import numpy as np
from analytical_timo_ring import ring_chain, wall_iso, wall_ply45

R = 0.0715
for name, mk in (("iso", lambda h: wall_iso(70e9, 0.3, h)), ("[-45]", wall_ply45)):
    for hf in (0.5, 0.1):
        h = R * hf
        A3, D3, G2 = mk(h)
        d, info = ring_chain(A3, D3, G2, R)
        At = np.zeros((2, 2))
        idx = [0, 2]
        for a in range(2):
            for b in range(2):
                At[a, b] = A3[idx[a], idx[b]] - A3[idx[a], 1] * A3[idx[b], 1] / A3[1, 1]
        Dt11 = D3[0, 0] - D3[0, 1] ** 2 / D3[1, 1]
        X = info["X"]
        print("%s h/R=%.2f:" % (name, hf))
        print("  EA / (2piR At11)                = %.6f" % (d[0] / (2 * np.pi * R * At[0, 0])))
        print("  GJ / (2piR^3 At66)              = %.6f" % (d[3] / (2 * np.pi * R ** 3 * At[1, 1])))
        print("  C14 / (2piR^2 At16)             = %.6f" % (X[0, 1] / (2 * np.pi * R ** 2 * At[0, 1] + 1e-300)))
        print("  GA / (piR At66 (1+h^2/6R^2))    = %.6f" % (d[1] / (np.pi * R * At[1, 1] * (1 + hf ** 2 / 6))))
        print("  EI / (piR^3 At11 + piR Dt11)    = %.6f" % (d[4] / (np.pi * R ** 3 * At[0, 0] + np.pi * R * Dt11)))
