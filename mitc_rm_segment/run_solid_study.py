"""Solid FEniCS runner for the taper convergence study (run in WSL opensg_env_v8
from OpenSG-1.0/examples).  Computes boundary (Taper=False) + taper (Taper=True)
6x6 for every solid mesh in out/taper_study/meshes and saves next to the RM
results:  solid_<tag>_{L,R,seg}.npy  + timing lines in solid_timings.txt."""
import os, sys, glob, time
# opensg (solid) package location: env var OPENSG_SOLID_PATH, else the WSL default.
sys.path.insert(0, os.environ.get("OPENSG_SOLID_PATH", "/mnt/c/Users/bagla0/OpenSG-1.0"))
import numpy as np
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness

# repo mitc_rm_segment dir: derived from this file so it works on WSL and the SSH server.
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = sys.argv[2] if len(sys.argv) > 2 else "taper_study"   # out/<STUDY>/{meshes,results}
MESH = HERE + "/out/" + STUDY + "/meshes"
RES = HERE + "/out/" + STUDY + "/results"
os.makedirs(RES, exist_ok=True)

only = sys.argv[1] if len(sys.argv) > 1 else ""            # substring filter, e.g. 'thin_iso'
log = open(RES + "/solid_timings.txt", "a")
for yf in sorted(glob.glob(MESH + "/solid_*.yaml")):
    tag = os.path.basename(yf)[6:-5]
    if only and only not in tag:
        continue
    if os.path.exists(RES + "/solid_%s_seg.npy" % tag):
        print("skip", tag); continue
    t0 = time.time()
    sm = SolidSegmentMesh(yf)
    mp, dens = sm.material_database
    bnd = np.asarray(compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)[0])
    tb = time.time() - t0
    t1 = time.time()
    tap = np.asarray(compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=True)[0])
    tt = time.time() - t1
    np.save(RES + "/solid_%s_L.npy" % tag, bnd[0])
    np.save(RES + "/solid_%s_R.npy" % tag, bnd[1])
    np.save(RES + "/solid_%s_seg.npy" % tag, tap)
    line = "solid %-20s boun %.0fs taper %.0fs  EA(L/R/seg) %.3f/%.3f/%.3f x1e9" % (
        tag, tb, tt, bnd[0][0, 0] / 1e9, bnd[1][0, 0] / 1e9, tap[0, 0] / 1e9)
    print(line); log.write(line + "\n"); log.flush()
    sys.stdout.flush()
