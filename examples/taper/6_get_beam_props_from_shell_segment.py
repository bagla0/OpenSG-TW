"""6_get_beam_props_from_shell_segment.py -- Timoshenko 6x6 of a 3-D SHELL tapered
segment with OpenSG-TW's JAX MITC-RM homogenizer (all-6-DOF independent-omega3
element, full transverse-shear integration; the settled production scheme).

Reads shell_segment.yaml, runs the tapered-segment solve (which also returns the two
end-ring 6x6), and prints DOF, wall time, and the 6x6.  Saves shell_segment_timo.npz.

Run (server opensg_2_0):
    python examples/taper/6_get_beam_props_from_shell_segment.py <shell_segment.yaml>
"""
import os
import shutil
import sys

import numpy as np
import yaml

from taper_common import print_timo, sym, Timer
import run_indep


def _nnodes(path):
    d = yaml.load(open(path), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return len(d["nodes"])


def main(seg_yaml):
    seg_yaml = os.path.abspath(seg_yaml)
    d = os.path.dirname(seg_yaml)
    work = os.path.join(d, "_shellrun")
    os.makedirs(work, exist_ok=True)
    tag = "seg"
    shutil.copy(seg_yaml, os.path.join(work, "shell_%s.yaml" % tag))
    with Timer() as t:
        r = run_indep.shell_solve_lagrange_sparse(tag, work, work, shear="full", return_full=True)
    dof = 6 * _nnodes(seg_yaml)                            # 6-DOF (independent omega_3) RM element
    S = print_timo("SHELL segment (OpenSG-TW JAX MITC-RM 3-D)  [%s]" % os.path.basename(seg_yaml),
                   r["S6"], dof, t.dt,
                   extra="extract %.1f + rings %.1f + seg %.1f s"
                   % (r["t_extract"], r["t_rings"], r["t_seg"]))
    np.savez(os.path.join(d, "shell_segment_timo.npz"), S6=S, C6L=sym(r["C6L"]),
             C6R=sym(r["C6R"]), dof=dof, time=t.dt)
    return S


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out",
                                                            "shell_segment.yaml"))
