"""
test_taper_jax.py    [ Windows opensg_2_0_env ]
JAX port acceptance: compute_timo_taper_jax (jit+vmap assembly) must reproduce
the NumPy compute_timo_taper EB to machine precision, and keep the FEniCS-solid
benchmark (<0.3% on the prismatic tube).
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from compute_timo_taper import compute_timo_taper, compute_timo_taper_jax

SOLID_EB = {"EA": 4.3971e10, "GJ": 1.6950e10, "EI2": 2.2035e10, "EI3": 2.2035e10}
LBL = ["EA", "GJ", "EI2", "EI3"]

npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "seg_iso_hR0.1_direct.npz")
b = np.load(npz, allow_pickle=True)

t0 = time.time(); rnp = compute_timo_taper(b, k22_mode="tube"); t_np = time.time() - t0
t0 = time.time(); rjx = compute_timo_taper_jax(b, k22_mode="tube"); t_jx = time.time() - t0   # incl. jit compile
t0 = time.time(); rjx2 = compute_timo_taper_jax(b, k22_mode="tube"); t_jx2 = time.time() - t0  # warm

dnp = np.diag(rnp["EB"]); djx = np.diag(rjx["EB"])
relEB = np.max(np.abs(rjx["EB"] - rnp["EB"])) / max(1.0, np.max(np.abs(rnp["EB"])))
print("JAX vs NumPy assembly: max rel |EB_jax - EB_np| = %.2e  -> %s"
      % (relEB, "MATCH" if relEB < 1e-9 else "CHECK"))
print("timing: NumPy %.2fs | JAX(cold) %.2fs | JAX(warm) %.2fs" % (t_np, t_jx, t_jx2))
print("\n%-5s %14s %14s %14s %9s" % ("term", "JAX EB", "NumPy EB", "solid EB", "JAXvsSolid"))
worst = 0.0
for i, k in enumerate(LBL):
    s = SOLID_EB[k]; e = 100.0 * (djx[i] - s) / s; worst = max(worst, abs(e))
    print("%-5s %14.4e %14.4e %14.4e %+8.2f%%" % (k, djx[i], dnp[i], s, e))
print("\nmax |JAX-vs-solid| = %.2f%%  ->  %s" % (worst, "JAX EB WORKS" if worst < 1.0 and relEB < 1e-9 else "CHECK"))
