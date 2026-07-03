"""Solid FEniCS runner for the taper convergence study (run in WSL opensg_env_v8
from OpenSG-1.0/examples).  Computes boundary (Taper=False) + taper (Taper=True)
6x6 for every solid mesh in out/taper_study/meshes and saves next to the RM
results:  solid_<tag>_{L,R,seg}.npy  + timing lines in solid_timings.txt."""
import os, sys, glob, time
sys.path.insert(0, "/mnt/c/Users/bagla0/OpenSG-1.0")
import numpy as np
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness

HERE = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/mitc_rm_segment"
MESH = HERE + "/out/taper_study/meshes"
RES = HERE + "/out/taper_study/results"
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
