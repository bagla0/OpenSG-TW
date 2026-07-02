import sys, time
sys.path.insert(0, "/mnt/c/Users/bagla0/OpenSG-1.0")
import numpy as np
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun
y = "/mnt/c/Users/bagla0/AppData/Local/Temp/claude/C--Users-bagla0/91cf4f05-ed42-47e2-974c-813d98a91247/scratchpad/annulus_tube.yaml"
t0 = time.time()
sm = SolidBounMesh(y)
mp, dens = sm.material_database
timo = compute_timo_boun(mp, sm.meshdata)[0]
np.set_printoptions(precision=4, suppress=False, linewidth=160)
print("FEniCS SOLID annulus Timoshenko 6x6:\n", np.asarray(timo))
d = np.diag(np.asarray(timo))
print("\nsolid EB-diagonal [EA, GA2, GA3, GJ, EI2, EI3]:", np.array2string(d, precision=4))
print("solid EB [EA, GJ, EI2, EI3] = [%.4e, %.4e, %.4e, %.4e]" % (d[0], d[3], d[4], d[5]))
print("time %.1fs" % (time.time() - t0))
