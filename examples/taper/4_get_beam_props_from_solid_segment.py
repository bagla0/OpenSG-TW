"""4_get_beam_props_from_solid_segment.py -- Timoshenko 6x6 of a 3-D SOLID tapered
segment with OpenSG-FEniCS (the `opensg` package, merged into this env).

Reads solid_segment.yaml, runs the 3-D MSG solid homogenization (Taper=True, so it also
returns the two end-ring 6x6), and prints DOF, wall time, and the 6x6.  Saves
solid_segment_timo.npz next to the YAML.

Run (server opensg_2_0):
    python examples/taper/4_get_beam_props_from_solid_segment.py <solid_segment.yaml>
"""
import os
import sys

import numpy as np

from taper_common import print_timo, sym, Timer
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness


def main(seg_yaml):
    seg_yaml = os.path.abspath(seg_yaml)
    d = os.path.dirname(seg_yaml)
    os.chdir(d)                                            # gmsh scratch (SG_mesh.msh) local
    sm = SolidSegmentMesh(seg_yaml)
    mp, _den = sm.material_database
    with Timer() as t:
        S, V0, V1s, DL, DR = compute_stiffness(mp, sm.meshdata, sm.left_submesh,
                                               sm.right_submesh, Taper=True)
    dof = 3 * sm.num_nodes                                 # 3-D displacement field
    S = print_timo("SOLID segment (OpenSG-FEniCS 3-D)  [%s]" % os.path.basename(seg_yaml),
                   S, dof, t.dt, extra="%d hexes" % sm.num_elements)
    np.savez(os.path.join(d, "solid_segment_timo.npz"), S6=S, C6L=sym(DL), C6R=sym(DR),
             dof=dof, time=t.dt)
    return S


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out",
                                                            "solid_segment.yaml"))
