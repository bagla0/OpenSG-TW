"""5_get_beam_props_from_shell_boundary.py -- Timoshenko 6x6 of a 1-D SHELL boundary
cross-section (ring) with OpenSG-TW's JAX MITC-RM homogenizer.

Reads a shell_boundary_{L,R}.yaml (1-D contour on the OML) and computes the ring 6x6 at
the OML reference (frac=0.0, curved), MITC transverse shear.  Prints DOF, wall time, 6x6.

Run (server opensg_2_0):
    python examples/taper/5_get_beam_props_from_shell_boundary.py <shell_boundary_L.yaml>
"""
import os
import sys

import numpy as np
import yaml

from taper_common import print_timo, Timer
from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6


def _nnodes(path):
    d = yaml.load(open(path), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return len(d["nodes"])


def main(boun_yaml):
    boun_yaml = os.path.abspath(boun_yaml)
    with Timer() as t:
        C = rm_timoshenko_6x6(boun_yaml, 0.0, curved=True, shear="mitc", orient=False)
    dof = 5 * _nnodes(boun_yaml)                           # 5-DOF/node C0 RM line element
    S = print_timo("SHELL boundary (OpenSG-TW JAX MITC-RM 1-D)  [%s]" % os.path.basename(boun_yaml),
                   C, dof, t.dt)
    np.savez(os.path.join(os.path.dirname(boun_yaml),
                          os.path.basename(boun_yaml).replace(".yaml", "_timo.npz")),
             C6=S, dof=dof, time=t.dt)
    return S


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out",
                                                            "shell_boundary_L.yaml"))
