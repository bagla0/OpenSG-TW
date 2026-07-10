"""taper_common.py -- shared setup + helpers for the tapered-segment example pipeline.

One env runs everything (server `opensg_2_0`: jax + dolfinx-0.8 + windIO + opensg):
  * mesh generation    -> opensg_io (third_party/OpenSG_io)
  * SHELL Timoshenko   -> OpenSG-TW JAX MITC-RM (opensg_jax + mitc_rm_segment)
  * SOLID Timoshenko   -> OpenSG-FEniCS (the `opensg` package, merged into this env)

All paths resolve relative to the OpenSG-TW repo root, so the pipeline is reproducible
from a fresh clone once examples/data/windio/IEA-22-280-RWT.yaml is present.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(os.path.dirname(HERE))                 # OpenSG-TW repo root
for p in (os.path.join(TW, "third_party", "OpenSG_io"),
          os.path.join(TW, "opensg_jax"),
          os.path.join(TW, "mitc_rm_segment"),
          TW):
    if p not in sys.path:
        sys.path.insert(0, p)

WINDIO = os.path.join(TW, "examples", "data", "windio", "IEA-22-280-RWT.yaml")
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def blade_span_z(blade, r, L_default=137.0):
    """Physical span position z(r) from the windIO reference axis (fallback r*L)."""
    try:
        ra = blade.osh["reference_axis"]["z"]
        return float(np.interp(r, ra["grid"], ra["values"]))
    except Exception:
        return r * L_default


def sym(M):
    M = np.asarray(M, float)
    return 0.5 * (M + M.T)


def print_timo(title, S, dof, dt, extra=""):
    """Standard Timoshenko-run report: DOF, wall time, and the symmetric 6x6."""
    S = sym(S)
    print("\n" + "=" * 72)
    print("%s" % title)
    print("  DOF used : %d" % int(dof))
    print("  time     : %.2f s%s" % (dt, ("   " + extra) if extra else ""))
    print("  Timoshenko 6x6  [EA, GA2, GA3, GJ, EI2, EI3]:")
    for i in range(6):
        print("    " + "".join("%14.5e" % S[i, j] for j in range(6)))
    print("  diagonal:", "  ".join("%s=%.4e" % (LBL[i], S[i, i]) for i in range(6)))
    return S


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
