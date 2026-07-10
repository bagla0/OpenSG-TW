"""3_get_beam_props_from_solid_boundary.py -- Timoshenko 6x6 of a 2-D SOLID boundary
cross-section with OpenSG-FEniCS.

Reads a solid_boundary_{L,R}.yaml (2-D quad section), runs the cross-sectional MSG
solid homogenization (compute_timo_boun), and prints DOF, wall time, and the 6x6.

Run (server opensg_2_0):
    python examples/taper/3_get_beam_props_from_solid_boundary.py <solid_boundary_L.yaml>
"""
import os
import sys

import numpy as np

from taper_common import print_timo, Timer
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun


def main(boun_yaml):
    boun_yaml = os.path.abspath(boun_yaml)
    d = os.path.dirname(boun_yaml)
    os.chdir(d)
    sm = SolidBounMesh(boun_yaml)
    mp, _den = sm.material_database
    with Timer() as t:
        Deff, V0, V1 = compute_timo_boun(mp, sm.meshdata)
    dof = 3 * sm.num_nodes
    S = print_timo("SOLID boundary (OpenSG-FEniCS 2-D)  [%s]" % os.path.basename(boun_yaml),
                   Deff, dof, t.dt, extra="%d quads" % sm.num_elements)
    np.savez(os.path.join(d, os.path.basename(boun_yaml).replace(".yaml", "_timo.npz")),
             C6=S, dof=dof, time=t.dt)
    return S


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out",
                                                            "solid_boundary_L.yaml"))
