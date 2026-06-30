"""Quick test: does moving the web from its baseline (x=0) to its true centre
(x=-t/2) change the homogenised Timoshenko stiffness?  Iso thick case."""
import os
import sys
import numpy as np
import yaml

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)
from gradient_kirchhoff import gradient_junction_kirchhoff
from strip_RM import rm_timoshenko_6x6

D = os.path.join(CC, "multicell_tube", "data")
T, R = 0.016, 0.05
xc = -T / 2.0
TERMS = [("EA", 0, 0), ("GA2", 1, 1), ("GA3", 2, 2), ("GJ", 3, 3),
         ("EI2", 4, 4), ("EI3", 5, 5), ("C14", 0, 3)]


def parse(r):
    s = r[0] if isinstance(r, list) else r
    return [float(v) for v in str(s).split()]


# build web-centred mesh
d = yaml.safe_load(open(os.path.join(D, "tube2cell_thick.yaml")))
N = [parse(r) for r in d["nodes"]]
nshift = 0
out = []
for (x, y, z) in N:
    if abs(x) < 1e-9 and abs(y) < R - 1e-6:      # interior web node
        x = xc
        nshift += 1
    out.append(["%.10f %.10f %.10f" % (x, y, z)])
d["nodes"] = out
wc = os.path.join(D, "tube2cell_thick_webc.yaml")
yaml.dump(d, open(wc, "w"), sort_keys=False, default_flow_style=False)
print("shifted %d interior web nodes to x=%.4f" % (nshift, xc))

S = np.loadtxt(os.path.join(D, "C6_solid_tube2cell_thick.txt"))
S = 0.5 * (S + S.T)


def errs(mesh):
    KL = np.asarray(gradient_junction_kirchhoff(mesh, frac=0.0, dshift=T / 2.0)[0])
    KL = 0.5 * (KL + KL.T)
    RM = np.asarray(rm_timoshenko_6x6(mesh, 0.0, dshift=T / 2.0, curved=True))
    RM = 0.5 * (RM + RM.T)
    return KL, RM


def pe(M, i, j):
    return 100.0 * (M[i, j] - S[i, j]) / S[i, j]


KL0, RM0 = errs(os.path.join(D, "tube2cell_thick.yaml"))
KL1, RM1 = errs(wc)
print("\n%-6s | %18s | %18s" % ("term", "KL  (x=0 -> x=-t/2)", "RM  (x=0 -> x=-t/2)"))
for (lab, i, j) in TERMS:
    print("%-6s | %+7.2f -> %+7.2f   | %+7.2f -> %+7.2f"
          % (lab, pe(KL0, i, j), pe(KL1, i, j), pe(RM0, i, j), pe(RM1, i, j)))
