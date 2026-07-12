"""compare_st15_schemes.py -- st15 Timoshenko 6x6 %diff vs the 2-D solid (VABS .K), for the
RM ring under three shear-tying schemes (full / tie-only-gamma23 / tie-both) and the old KL
Hermite shell.  Confirms that tying gamma13 is unnecessary on the prismatic ring (gamma13 is
algebraic in the directors -- no differentiated-displacement/rotation pairing that locks), so
'mitc4_g23' (tie only gamma23) == 'full', and both are the production scheme."""
import os, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment")); sys.path.insert(0, MITC)
import jax; jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml
from oml_ring import load_ring_ref
from run_ring_indep import ring_indep

BENCH = os.path.join(HERE, "..", "..", "..", "examples", "data", "benchmark")
SHELL = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
def load_K(p):
    L = open(p).read().splitlines()
    i = next(k for k, ln in enumerate(L) if "Timoshenko Stiffness Matrix" in ln)
    rows = []
    for ln in L[i+1:]:
        s = ln.split()
        try: [float(x) for x in s]; ok = len(s) == 6
        except ValueError: ok = False
        if ok: rows.append([float(x) for x in s])
        if len(rows) == 6: break
    return np.array(rows)
K = load_K(os.path.join(BENCH, "st15_vabs.K")); K = 0.5*(K+K.T)
lbl = ["EA","GA2","GA3","GJ","EI2","EI3"]

R = load_ring_ref(SHELL, "oml")
def ring(sch):
    C = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                   R["k22"], R["ax"], R["cross"], shear=sch, lam_space="elem")
    return 0.5*(C+C.T)
KL = np.asarray(solve_tw_from_yaml(SHELL, frac=0.0)["Timo"]); KL = 0.5*(KL+KL.T)
models = [("KL-Hermite (old)", KL),
          ("RM full", ring("full")),
          ("RM tie-g23 only", ring("mitc4_g23")),
          ("RM tie-both", ring("mitc4_both"))]

print("st15 Timoshenko 6x6 diagonal %% diff vs 2-D solid (VABS .K)\n")
print("  %-18s %6s %6s %6s %6s %6s %6s   %s" % ("model", *lbl, "Frob%"))
for nm, C in models:
    e = [100*(C[i,i]-K[i,i])/K[i,i] for i in range(6)]
    fro = np.linalg.norm(C-K)/np.linalg.norm(K)*100
    print("  %-18s %+6.1f %+6.1f %+6.1f %+6.1f %+6.1f %+6.1f   %5.2f" % (nm, *e, fro))

g23 = ring("mitc4_g23"); full = ring("full"); both = ring("mitc4_both")
print("\ngamma13-tying effect on the RING 6x6 (max |diag diff|, %):")
print("  full   vs tie-g23 : %.3f%%" % max(abs(100*(full[i,i]-g23[i,i])/full[i,i]) for i in range(6)))
print("  tie-g23 vs tie-both: %.3f%%" % max(abs(100*(both[i,i]-g23[i,i])/g23[i,i]) for i in range(6)))
print("  -> gamma13 tie changes the ring 6x6 by <~this; gamma13 needs no MITC on the prismatic ring.")
