import os, sys
import numpy as np
seg = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mitc_rm_segment"
sys.path.insert(0, seg)
from compute_timo_taper import compute_timo_taper
from solve_segment_jax import solve_boundary_bundle
OUT = r"C:\Users\bagla0\AppData\Local\Temp\claude\C--Users-bagla0\91cf4f05-ed42-47e2-974c-813d98a91247\scratchpad"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

b = np.load(os.path.join(seg, "out", "BAR_URC_numEl_52_segment_0.npz"), allow_pickle=True)
# shell boundary rings (1-D RM cross-sections)
rL = solve_boundary_bundle(b, "L", shear="mitc"); rR = solve_boundary_bundle(b, "R", shear="mitc")
# shell tapered segment 6x6
rseg = compute_timo_taper(b, k22_mode="general", return_timo=True)

shL = np.diag(rL["C6"]); shR = np.diag(rR["C6"]); shSeg = np.diag(rseg["C6"])
soL = np.diag(np.load(OUT + "/solid_seg0_boun_L.npy"))
soR = np.diag(np.load(OUT + "/solid_seg0_boun_R.npy"))
soSeg = np.diag(np.load(OUT + "/solid_seg0_6x6.npy"))

print("seg 0  --  diagonal Timoshenko terms (x1e9)\n")
print("%-5s | %8s %8s %8s | %8s %8s %8s | %8s %8s" %
      ("term", "shL", "soL", "err%", "shR", "soR", "err%", "shSeg", "soSeg"))
print("-" * 92)
for i, k in enumerate(LBL):
    eL = 100 * (shL[i] - soL[i]) / soL[i]; eR = 100 * (shR[i] - soR[i]) / soR[i]
    print("%-5s | %8.3f %8.3f %+7.1f | %8.3f %8.3f %+7.1f | %8.3f %8.3f" %
          (k, shL[i]/1e9, soL[i]/1e9, eL, shR[i]/1e9, soR[i]/1e9, eR, shSeg[i]/1e9, soSeg[i]/1e9))

print("\nshell taper ratio EA(R)/EA(L) = %.3f ; solid taper ratio = %.3f" %
      (shR[0]/shL[0], soR[0]/soL[0]))
print("shell seg vs shell-L: %+.1f%% | shell seg vs (L+R)/2: %+.1f%%" %
      (100*(shSeg[0]-shL[0])/shL[0], 100*(shSeg[0]-0.5*(shL[0]+shR[0]))/(0.5*(shL[0]+shR[0]))))
print("solid seg vs solid-L: %+.1f%% | solid seg vs (L+R)/2: %+.1f%%" %
      (100*(soSeg[0]-soL[0])/soL[0], 100*(soSeg[0]-0.5*(soL[0]+soR[0]))/(0.5*(soL[0]+soR[0]))))
