"""Step 2 - JAX Reissner-Mindlin (MITC, shear='mitc_both') Timoshenko 6x6, all cases.

Writes results/C6_RM_<case>.dat (symmetric 6x6, order [EA,GA2,GA3,GJ,EI2,EI3]).

  single-cell (smooth circle) : exact hoop curvature k22 = -1/R
                                (msg_rm_timo.timoshenko_rm)
  two-cell    (webbed tube)   : geometric per-element curvature (public driver
                                rm_timoshenko_6x6, curved=True)
"""
import os
import time

from common import cases, compute_rm, save_dat, RES

t0 = time.time()
print("=== Reissner-Mindlin (RM) Timoshenko 6x6 ===")
for c in cases():
    RM = compute_rm(c)
    out = os.path.join(RES, "C6_RM_%s.dat" % c["name"])
    save_dat(out, RM, "JAX Reissner-Mindlin (MITC mitc_both) -- %s  [%s k22]"
             % (c["name"], c["method"]))
    print("[RM] %-18s  EA=%12.5e  GA2=%12.5e  GJ=%12.5e   (%4.0fs)"
          % (c["name"], RM[0, 0], RM[1, 1], RM[3, 3], time.time() - t0), flush=True)
print("RM done -> %s" % RES)
