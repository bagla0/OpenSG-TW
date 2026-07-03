import sys, time
sys.path.insert(0, "/mnt/c/Users/bagla0/OpenSG-1.0")
import numpy as np
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
OUT = "/mnt/c/Users/bagla0/AppData/Local/Temp/claude/C--Users-bagla0/91cf4f05-ed42-47e2-974c-813d98a91247/scratchpad"
for segid in [int(a) for a in sys.argv[1:]]:
    t0 = time.time()
    yf = "/mnt/c/Users/bagla0/OpenSG-1.0/examples/data/Solid_3DSG/bar_urc_npl_1_ar_5-segment_%d.yaml" % segid
    sm = SolidSegmentMesh(yf)
    mp, dens = sm.material_database
    # axis + node span from the mesh geometry
    xg = sm.meshdata["mesh"].geometry.x
    e1 = np.mean([np.asarray(f.x.array).reshape(-1, 3) for f in [sm.meshdata["frame"][0]]][0], axis=0)
    ax = int(np.argmax(np.abs(e1)))
    lo, hi = float(xg[:, ax].min()), float(xg[:, ax].max())
    res = list(compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False))
    bnd = np.asarray(res[0])                       # (2,6,6)
    dL, dR = np.diag(bnd[0]), np.diag(bnd[1])
    np.save(OUT + "/solid_seg%d_boun_L.npy" % segid, bnd[0])
    np.save(OUT + "/solid_seg%d_boun_R.npy" % segid, bnd[1])
    print("seg %2d  origin=%.3f  axis=%d  span=[%.3f,%.3f]  (%.0fs)" % (segid, float(sm.meshdata["origin"]), ax, lo, hi, time.time() - t0))
    print("   Deff_L diag(x1e9) [EA,GA2,GA3,GJ,EI2,EI3]=%s" % np.array2string(dL/1e9, precision=3))
    print("   Deff_R diag(x1e9)                        =%s" % np.array2string(dR/1e9, precision=3))
    sys.stdout.flush()
