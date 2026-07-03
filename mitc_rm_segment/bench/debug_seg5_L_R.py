import os, sys
import numpy as np
seg = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mitc_rm_segment"
sys.path.insert(0, seg)
from compute_timo_taper import compute_timo_taper
from solve_segment_jax import solve_boundary_bundle
OUT = r"C:\Users\bagla0\AppData\Local\Temp\claude\C--Users-bagla0\91cf4f05-ed42-47e2-974c-813d98a91247\scratchpad"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

# shell seg_5 (span [17.24,20.69]) <-> solid seg_4 (same span)
b = np.load(os.path.join(seg, "out", "BAR_URC_numEl_52_segment_5.npz"), allow_pickle=True)
nd = np.asarray(b["seg_x"]); ax = int(b["axis"])
print("shell seg_5 span [%.3f, %.3f]  origin %.3f" % (nd[:, ax].min(), nd[:, ax].max(), nd[:, ax].mean()))
rL = solve_boundary_bundle(b, "L", shear="mitc"); rR = solve_boundary_bundle(b, "R", shear="mitc")
rseg = compute_timo_taper(b, k22_mode="general", return_timo=True)
shL, shR, shSeg = np.diag(rL["C6"]), np.diag(rR["C6"]), np.diag(rseg["C6"])
soL = np.diag(np.load(OUT + "/solid_seg4_boun_L.npy"))     # solid seg_4 L (x=17.24)
soR = np.diag(np.load(OUT + "/solid_seg4_boun_R.npy"))     # solid seg_4 R (x=20.69)

print("\nshell seg_5  vs  solid seg_4  (SAME span [17.24,20.69])  --  diag (x1e9)\n")
print("%-5s | %8s %8s %8s | %8s %8s %8s" % ("term", "shL", "soL(seg4)", "err%", "shR", "soR(seg4)", "err%"))
print("-" * 74)
for i, k in enumerate(LBL):
    eL = 100*(shL[i]-soL[i])/soL[i]; eR = 100*(shR[i]-soR[i])/soR[i]
    print("%-5s | %8.3f %8.3f %+7.1f | %8.3f %8.3f %+7.1f" %
          (k, shL[i]/1e9, soL[i]/1e9, eL, shR[i]/1e9, soR[i]/1e9, eR))
print("\nshell seg taper EA(R)/EA(L) = %.3f ; solid seg4 taper = %.3f" % (shR[0]/shL[0], soR[0]/soL[0]))
print("shell seg_5 6x6 diag =", np.array2string(shSeg/1e9, precision=3))
