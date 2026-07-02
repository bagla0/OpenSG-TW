"""
bench_eb_vs_solid.py    [ Windows opensg_2_0_env ]
========================================================================
EB benchmark the user asked for: the JAX MITC-RM shell tapered-segment EB
(compute_timo_taper) vs the FEniCS SOLID (OpenSG-1.0 compute_timo_boun) on the
SAME prismatic tube (the controllable, same-geometry case; for a prismatic
segment the origin is irrelevant since EB is per unit length).

Result (R=1, t=0.1, E=70 GPa, nu=0.3):
    term    shell EB (JAX)   solid EB (FEniCS)   diff
    EA      4.3979e10        4.3971e10          +0.02%
    GJ      1.6997e10        1.6950e10          +0.28%
    EI2     2.2003e10        2.2035e10          -0.15%
    EI3     2.2003e10        2.2035e10          -0.15%
=> EB is working (<0.3% on every term).

To (re)produce the SOLID reference (WSL, opensg_env_v8 + OpenSG-1.0):
    python bench/gen_annulus_solid.py            # writes annulus_tube.yaml
    wsl ... python3 bench/run_solid_annulus_fenics.py   # -> FEniCS solid 6x6
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, SEG)
from compute_timo_taper import compute_timo_taper

# FEniCS solid annulus reference 6x6 diagonal (opensg_env_v8, run_solid_annulus_fenics.py)
SOLID6 = [4.3971e10, 8.4920e9, 8.4920e9, 1.6950e10, 2.2035e10, 2.2035e10]
LBL6 = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

npz = os.path.join(SEG, "out", "seg_iso_hR0.1_direct.npz")
b = np.load(npz, allow_pickle=True)
r = compute_timo_taper(b, k22_mode="tube", return_timo=True)
d6 = np.diag(r["C6"])
print("Prismatic tube  R=%.2f  L=%.2f  origin=%.2f" % (r["R"], r["L"], r["origin"]))
print("%-5s %16s %16s %9s" % ("term", "shell (JAX RM)", "solid (FEniCS)", "diff"))
worst = 0.0
for i, k in enumerate(LBL6):
    s = SOLID6[i]; e = 100.0 * (d6[i] - s) / s; worst = max(worst, abs(e))
    print("%-5s %16.4e %16.4e %+8.2f%%" % (k, d6[i], s, e))
print("\nmax |diff| = %.2f%%  ->  %s" % (worst,
      "TIMO 6x6 WORKS (shell == solid, incl. transverse shear)" if worst < 1.0 else "CHECK"))
