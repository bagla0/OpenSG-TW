"""time_solid_cases.py -- fresh wall-clock timing of the 8 solid benchmark cases
with the optimized FEniCS solver (same two-call convention as the original
benchmark timings: boun = Taper=False call, taper = Taper=True call)."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("taper_square", "square", r, m) for r in ("thin", "thick") for m in ("iso", "m45")] \
      + [("taper_study", "circle", r, m) for r in ("thin", "thick") for m in ("iso", "m45")]

print("%-22s %8s %8s %8s %8s" % ("case", "load", "boun", "taper", "total"))
for sub, geom, reg, mat in CASES:
    y = os.path.join(HERE, "out", sub, "meshes", "solid_%s_%s_aR070.yaml" % (reg, mat))
    t0 = time.time()
    sm = SolidSegmentMesh(y)
    t1 = time.time()
    mp, dens = sm.material_database
    compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)
    t2 = time.time()
    tap = np.asarray(compute_stiffness(mp, sm.meshdata, sm.left_submesh,
                                       sm.right_submesh, Taper=True)[0])
    t3 = time.time()
    print("%-22s %8.1f %8.1f %8.1f %8.1f   EA=%.4e"
          % ("%s-%s-%s" % (geom, reg, mat), t1 - t0, t2 - t1, t3 - t2, t3 - t1, tap[0, 0]))
    sys.stdout.flush()
