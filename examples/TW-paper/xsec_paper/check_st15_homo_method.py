"""Which model built Table 13? KL-Hermite (solve_tw_from_yaml) vs RM ring (ring_indep),
both vs VABS .K, for the st15 shell yaml(s).  Prints diag %err so we can see whether the
paper's st15 homogenization table is RM-consistent with the dehom."""
import os, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml
from oml_ring import load_ring_ref, c6

BENCH = os.path.join(HERE, "..", "..", "..", "examples", "data", "benchmark")
def load_K(p):
    L = open(p).read().splitlines()
    i = next(k for k, ln in enumerate(L) if "Timoshenko Stiffness Matrix" in ln)
    rows = []
    for ln in L[i+1:]:
        s = ln.split()
        try:
            [float(x) for x in s]; ok = len(s) == 6
        except ValueError:
            ok = False
        if ok: rows.append([float(x) for x in s])
        if len(rows) == 6: break
    return np.array(rows)
K = load_K(os.path.join(BENCH, "st15_vabs.K")); K = 0.5*(K+K.T)
lbl = ["EA","GA2","GA3","GJ","EI2","EI3"]
YAMLS = {"examples/st15_shell.yaml": os.path.join(HERE,"..","..","..","examples","data","1d_yaml","st15_shell.yaml"),
         "tests/1Dshell_15.yaml": os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")}
print("VABS .K diag:", " ".join("%s=%.3e"%(lbl[i],K[i,i]) for i in range(6)))
for nm, y in YAMLS.items():
    if not os.path.exists(y): print(nm, "MISSING"); continue
    kl = np.asarray(solve_tw_from_yaml(y, frac=0.0)["Timo"]); kl = 0.5*(kl+kl.T)
    rm = c6(load_ring_ref(y, "oml"))
    print("\n== %s ==" % nm)
    print("  %-5s %12s %12s | %8s %8s" % ("term","KL-Herm","RM-ring","KL%err","RM%err"))
    for i in range(6):
        print("  %-5s %12.4e %12.4e | %+7.1f %+7.1f" %
              (lbl[i], kl[i,i], rm[i,i], 100*(kl[i,i]-K[i,i])/K[i,i], 100*(rm[i,i]-K[i,i])/K[i,i]))
    print("  Frobenius vs .K:  KL %.2f%%   RM %.2f%%" %
          (np.linalg.norm(kl-K)/np.linalg.norm(K)*100, np.linalg.norm(rm-K)/np.linalg.norm(K)*100))
