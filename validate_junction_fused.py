"""Validate gradient_kirchhoff_fused against gradient_junction_kirchhoff.

The fused path folds the junction tie-constraint into the local->global assembly
(weighted scatter), instead of building the broken matrices and projecting with
A_g = T^T A_b T.  It must reproduce the 6x6 stiffness to machine precision.

Run:  python validate_junction_fused.py
"""
import os, sys, time
CC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CC)
import numpy as np
from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
from opensg_jax.fe_jax.gradient_kirchhoff_fused import gradient_junction_kirchhoff_fused

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def run():
    IB = os.path.join(CC, "examples", "data", "iea_blade")
    ymls = [("iea_r050", os.path.join(IB, "shell_r050.yaml")),
            ("iea_r080", os.path.join(IB, "shell_r080.yaml"))]
    allpass = True
    for name, yml in ymls:
        if not os.path.exists(yml):
            print("SKIP (missing):", yml); continue
        # warm the JIT (vmap recompiles per element count), then time the warm calls
        C_o, nj_o, ng_o = gradient_junction_kirchhoff(yml, frac=0.0, orient=False)
        C_f, nj_f, ng_f = gradient_junction_kirchhoff_fused(yml, frac=0.0, orient=False)
        t0 = time.time(); C_o, _, _ = gradient_junction_kirchhoff(yml, frac=0.0, orient=False); t1 = time.time()
        C_f, _, _ = gradient_junction_kirchhoff_fused(yml, frac=0.0, orient=False); t2 = time.time()
        ad = float(np.max(np.abs(C_f - C_o)))
        scale = float(np.max(np.abs(C_o)))
        rel_to_max = ad / (scale + 1e-30)
        ok = rel_to_max < 1e-9
        allpass &= ok
        print("\n=== %s  (%s) ===" % (name, "PASS" if ok else "FAIL"))
        print("junctions orig=%d fused=%d | ng orig=%d fused=%d" % (nj_o, nj_f, ng_o, ng_f))
        print("warm time orig=%.2fs  fused=%.2fs" % (t1 - t0, t2 - t1))
        print("max abs diff=%.3e  abs/max-term=%.3e" % (ad, rel_to_max))
        do, df = np.diag(C_o), np.diag(C_f)
        for i in range(6):
            print("  %-4s orig=% .6e  fused=% .6e  rel=%.2e" %
                  (LBL[i], do[i], df[i], abs(df[i] - do[i]) / (abs(do[i]) + 1e-30)))
    print("\n==== ALL PASS ====" if allpass else "\n==== SOME FAIL ====")
    return allpass


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
