"""Is the C13 (ext-bend2) '2x' a formulation factor-2 or cancellation of a
near-zero coupling?  Compare JAX EB vs VABS .K for st0 AND st15, both C13 (small)
and C14 (ext-bend3, large).  A formulation factor-2 would hit C13 AND C14
equally on BOTH sections; cancellation hits only the near-zero coupling."""
import re, sys, numpy as np
sys.path.insert(0, r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\opensg_jax")
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data\opensg-FEniCS\data"
YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG"


def vabs_classical(kfile):
    """parse the 4x4 Classical Stiffness [ext,twist,bend2,bend3] + tension center."""
    txt = open(kfile).read().splitlines()
    i = next(i for i, l in enumerate(txt) if "Classical Stiffness Matrix" in l)
    rows = []
    for l in txt[i + 3:i + 7]:
        rows.append([float(v) for v in l.split()])
    C = np.array(rows)
    j = next(k for k, l in enumerate(txt) if "Tension Center" in l)
    tc = [float(v) for v in txt[j + 3].split()]
    return C, tc


for st in (0, 15):
    K, tc = vabs_classical(rf"{DATA}\bar_urc-{st}-t-0.in.K")
    EB, _, _ = timoshenko_from_yaml(rf"{YAML}\1Dshell_{st}.yaml", frac=0.0)
    EB = np.asarray(EB)
    EA = EB[0, 0]; EA_K = K[0, 0]
    print(f"\n=== station {st} ===  EA: JAX {EA:.4e}  VABS {EA_K:.4e}  ({100*(EA-EA_K)/EA_K:+.1f}%)")
    for (i, j), nm in [((0, 2), "C13 ext-bend2"), ((0, 3), "C14 ext-bend3")]:
        jv, kv = EB[i, j], K[i, j]
        ratio = jv / kv if abs(kv) > 1e-30 else float("nan")
        print(f"  {nm:14s}: JAX {jv:+.4e}  VABS {kv:+.4e}  ratio {ratio:+.3f}  "
              f"(JAX/EA={jv/EA:+.3e}, VABS/EA={kv/EA_K:+.3e})")
    # tension center from JAX: xt3 = C13/EA, xt2 = -C14/EA
    print(f"  tension center y3: JAX {EB[0,2]/EA*1e6:+.2f} um   VABS {tc[1]*1e6:+.2f} um  (drives C13)")
    print(f"  tension center y2: JAX {-EB[0,3]/EA*1e3:+.3f} mm  VABS {tc[0]*1e3:+.3f} mm  (drives C14)")
