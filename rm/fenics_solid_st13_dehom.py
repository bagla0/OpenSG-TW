"""FEniCS-solid dehom for station 13 (data/2Dsolid_12.yaml, beam_x=44.83=st13),
under a representative blade-section load. Saves the full quad-point material-frame
stress field + coords for the shell-vs-solid path comparison."""
import os, sys
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
WORK = "/tmp/st13"; os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
YML = PKG + "/data/2Dsolid_12.yaml"
OUTDIR = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/outputs/st12"
os.makedirs(OUTDIR, exist_ok=True)
FF = np.array([2.0e5, 0.0, 0.0, 3.0e5, 8.0e5, 5.0e5])   # [F1,F2,F3,M1(twist),M2,M3]

from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun
import opensg.core.stress_recov as stress_recov

sm = SolidBounMesh(YML)
mat_param, density = sm.material_database
timo = compute_timo_boun(mat_param, sm.meshdata)
beam_out = ([[[0, 0.0] for _ in range(6)] for _ in range(3)], None)
st_m, u_loc, strain_q, stress_q, coord_q = stress_recov.local_strain(
    timo, beam_out, 0, sm.meshdata, mat_param, FF)
coords = coord_q.reshape(-1, 3)                # [y1,y2,y3]
stress = stress_q.x.array.reshape(-1, 6)       # [S11,S22,S33,S23,S13,S12] material frame
full = np.column_stack([coords[:, 1], coords[:, 2], stress])
np.savetxt(os.path.join(OUTDIR, "st13_solid_dehom_full.txt"), full,
           header="y2 y3 S11 S22 S33 S23 S13 S12", comments="")
print(f"st13 solid dehom: {len(coords)} quad pts, FF={FF.tolist()}")
print(f"  y2 range [{coords[:,1].min():.3f},{coords[:,1].max():.3f}]  "
      f"max|S11|={np.max(np.abs(stress[:,0]))/1e6:.2f} MPa")
