import sys, time
sys.path.insert(0, "/mnt/c/Users/bagla0/OpenSG-1.0")
import numpy as np
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
OUT = "/mnt/c/Users/bagla0/AppData/Local/Temp/claude/C--Users-bagla0/91cf4f05-ed42-47e2-974c-813d98a91247/scratchpad"
yf = "/mnt/c/Users/bagla0/OpenSG-1.0/examples/data/Solid_3DSG/bar_urc_npl_1_ar_5-segment_0.yaml"
t0 = time.time()
sm = SolidSegmentMesh(yf)
mp, dens = sm.material_database
# Taper=False -> boundary cross-section 6x6 [Deff_l, Deff_r]
res = compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)
res = list(res)
np.set_printoptions(precision=4, linewidth=160)
print("Taper=False returned %d arrays  (%.1fs)" % (len(res), time.time() - t0))
bnd = np.asarray(res[0])                       # (2,6,6) = [Deff_l, Deff_r]
for k, nm in ((0, "L"), (1, "R")):
    A = bnd[k]; np.save(OUT + "/solid_seg0_boun_%s.npy" % nm, A)
    print("  Deff_%s 6x6 diag [EA,GA2,GA3,GJ,EI2,EI3] = %s" % (nm, np.array2string(np.diag(A), precision=4)))
print("origin:", float(sm.meshdata["origin"]))
