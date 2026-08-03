"""timo_ring_table.py -- analytical (closed-form eq:timo_tube) values for tab:single:
paper's [-45] tube, R=0.0715, R/h=2 and 10, vs the tabulated 2-D solid values.
Also tests a closed-form candidate for the shear-bending coupling C25=C36.
"""
import numpy as np
from analytical_timo_ring import ring_chain, wall_ply45

R = 0.0715
solid = {
    2:  {"C11": 1.9730e8, "C14": -4.0737e6, "C22": 6.8641e7, "C25": 2.1084e6,
         "C33": 6.8641e7, "C36": 2.1084e6, "C44": 6.7459e5, "C55": 5.4042e5, "C66": 5.4042e5},
    10: {"C11": 3.9311e7, "C14": -7.7255e5, "C22": 1.2036e7, "C25": 3.8697e5,
         "C33": 1.2036e7, "C36": 3.8697e5, "C44": 1.2274e5, "C55": 1.0076e5, "C66": 1.0076e5},
}

for Rh in (2, 10):
    h = R / Rh
    A3, D3, G2 = wall_ply45(h)
    At = np.zeros((2, 2)); idx = [0, 2]
    for a in range(2):
        for b in range(2):
            At[a, b] = A3[idx[a], idx[b]] - A3[idx[a], 1] * A3[idx[b], 1] / A3[1, 1]
    Dt11 = D3[0, 0] - D3[0, 1] ** 2 / D3[1, 1]
    Dt16 = D3[0, 2] - D3[0, 1] * D3[1, 2] / D3[1, 1]
    cf = {
        "C11": 2 * np.pi * R * At[0, 0],
        "C14": 2 * np.pi * R ** 2 * At[0, 1],
        "C22": np.pi * R * At[1, 1] * (1 + (h / R) ** 2 / 6),
        "C33": np.pi * R * At[1, 1] * (1 + (h / R) ** 2 / 6),
        "C44": 2 * np.pi * R ** 3 * At[1, 1],
        "C55": np.pi * R ** 3 * At[0, 0] + np.pi * R * Dt11,
        "C66": np.pi * R ** 3 * At[0, 0] + np.pi * R * Dt11,
    }
    d, info = ring_chain(A3, D3, G2, R)
    X = info["X"]
    print("== R/h=%d (h=%.5f) ==" % (Rh, h))
    print("  X order check: diag = %s" % np.array2string(np.diag(X), precision=4))
    # C25 candidate: pi R^2 Atilde16  (same moment-arm progression as C14)
    c25_analytic = None
    # locate the shear-bending coupling in X: VABS order (1..6) -> X[1,4] = C25
    try:
        c25_analytic = X[1, 4]
        print("  X[2,5] (C25 harmonic) = %.5e   /(piR^2 At16) = %.6f   /(piR^2 At16 (1+h^2/6R^2)) = %.6f"
              % (c25_analytic, c25_analytic / (np.pi * R ** 2 * At[0, 1]),
                 c25_analytic / (np.pi * R ** 2 * At[0, 1] * (1 + (h / R) ** 2 / 6))))
    except Exception as e:
        print("  C25 extract failed:", e)
    for k in ("C11", "C14", "C22", "C33", "C44", "C55", "C66"):
        s = solid[Rh][k]
        # match table sign convention (solid C14 negative): flip closed form if opposite sign
        v = cf[k] if cf[k] * s >= 0 else -cf[k]
        print("  %s: analytic %+.4e   solid %+.4e   err %+6.2f%%" % (k, v, s, 100 * (v - s) / abs(s)))
