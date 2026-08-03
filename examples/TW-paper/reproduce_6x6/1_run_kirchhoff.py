"""Step 1 - JAX Kirchhoff-Love (gradient / Hermite-C1) Timoshenko 6x6, all cases.

Writes results/C6_KL_<case>.dat (symmetric 6x6, order [EA,GA2,GA3,GJ,EI2,EI3]).

  single-cell (smooth circle) : exact hoop curvature k22 = -1/R
  two-cell    (webbed tube)   : geometric per-element curvature (public driver
                                gradient_junction_kirchhoff)
"""
import os
import time

from common import cases, compute_kl, save_dat, RES

t0 = time.time()
print("=== Kirchhoff-Love (KL) Timoshenko 6x6 ===")
for c in cases():
    KF = compute_kl(c)
    out = os.path.join(RES, "C6_KL_%s.dat" % c["name"])
    save_dat(out, KF, "JAX Kirchhoff-Love (gradient-Hermite C1) -- %s  [%s k22]"
             % (c["name"], c["method"]))
    print("[KL] %-18s  EA=%12.5e  GA2=%12.5e  GJ=%12.5e   (%4.0fs)"
          % (c["name"], KF[0, 0], KF[1, 1], KF[3, 3], time.time() - t0), flush=True)
print("KL done -> %s" % RES)
