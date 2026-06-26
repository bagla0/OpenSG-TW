"""FEniCS-solid Timoshenko 6x6 for station 12 from data/2Dsolid_12.yaml (WSL,
opensg_env_v8). Order [ext, shear2, shear3, twist, bend2, bend3]."""
import os, sys
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
WORK = "/tmp/st12"; os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
YML = PKG + "/data/2Dsolid_12.yaml"
OUTDIR = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/outputs/st12"
os.makedirs(OUTDIR, exist_ok=True)

from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

sm = SolidBounMesh(YML)
mat_param, density = sm.material_database
print(f"st12 solid mesh: {sm.num_nodes} nodes, {sm.num_elements} quads, "
      f"{len(mat_param)} material phases")
timo = compute_timo_boun(mat_param, sm.meshdata)
C6 = np.asarray(timo[0])
np.savetxt(os.path.join(OUTDIR, "solid_C6_st12.txt"), C6)
np.set_printoptions(precision=4, suppress=False, linewidth=140)
print("\nFEniCS-solid st12 Timoshenko 6x6 [ext, sh2, sh3, twist, bend2, bend3]:\n")
print(C6)
