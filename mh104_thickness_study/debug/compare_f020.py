"""f=0.2, OML, strict-connected mesh: % diff of every nonzero Timoshenko term for
JAX-Kirchhoff (C1 Hermite), JAX-RM (C0), and FEniCS-shell vs the 2D SOLID (= VABS)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LBL = ["ext", "sh2", "sh3", "tw", "b2", "b3"]
DIAG = {0: "EA", 1: "GA2", 2: "GA3", 3: "GJ", 4: "EI2", 5: "EI3"}


def sym(C):
    return 0.5 * (C + C.T)


solid = sym(np.loadtxt(os.path.join(RES, "C6_solid_f020.txt")))
jk = sym(np.loadtxt(os.path.join(HERE, "C6_jax_kirchhoff_f020.txt")))
jr = sym(np.loadtxt(os.path.join(HERE, "C6_jax_rm_f020.txt")))
fc = sym(np.loadtxt(os.path.join(HERE, "C6_fenics_connect_f020.txt")))
EA = abs(solid[0, 0])
methods = [("JAX-Kirch", jk), ("JAX-RM", jr), ("FEniCS", fc)]

print("mh104 f=0.2  OML  strict-connected mesh   (% diff vs 2D solid = VABS)\n")
print("%-9s %13s | %9s %9s %9s" % ("term", "solid", "JAX-Kirch", "JAX-RM", "FEniCS"))
print("-" * 58)
acc = {m[0]: [] for m in methods}
accd = {m[0]: [] for m in methods}
for i in range(6):
    for j in range(i, 6):
        v = solid[i, j]
        if abs(v) < 1e-3 * EA:                      # skip noise-floor (<0.1% of EA)
            continue
        nm = DIAG[i] if i == j else "%s-%s" % (LBL[i], LBL[j])
        row = "%-9s %+13.4e |" % (nm, v)
        for mn, M in methods:
            e = 100 * (M[i, j] - v) / v
            row += " %+8.1f%%" % e
            acc[mn].append(abs(e))
            if i == j:
                accd[mn].append(abs(e))
        print(row)
print("-" * 58)
print("%-9s %13s |" % ("mean|%| diag", "") + "".join(" %8.1f%%" % np.mean(accd[m[0]]) for m in methods))
print("%-9s %13s |" % ("mean|%| all", "") + "".join(" %8.1f%%" % np.mean(acc[m[0]]) for m in methods))
