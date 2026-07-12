"""sweep_rm_shear.py -- does the RECOVERY tying scheme make sigma13/sigma23 physical?
Cap-centre path, RM dehom, s2 tying in {mitc4_g23, mitc4_wonly, mitc4_both}, vs VABS."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ""))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm
DEH = os.path.join(HERE, "..", "..", "..", "examples", "data", "dehom_st15")
SHELL15 = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
d = np.loadtxt(os.path.join(DEH, "bar_urc-15-t-0.in.SM"))
sm_xy, sm_s = d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]
tree = cKDTree(sm_xy)
coords = np.loadtxt(os.path.join(DEH, "solid.lp_sparcap_center_thickness_015.coords"))[:, :2]
V = sm_s[tree.query(coords)[1]]
zmm = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))] * 1e3
B = dehom_rm.build_rm_bundle(SHELL15, ref="oml")

def pk(a): return a[np.argmax(np.abs(a))]
print("cap-centre: peak |S13|,|S23| (MPa)   VABS = %.4f , %.4f" % (abs(pk(V[:,4]))/1e6, abs(pk(V[:,3]))/1e6))
for sch in ("mitc4_g23", "mitc4_wonly", "mitc4_both"):
    S = np.asarray(dehom_rm.stress_at_points(B, coords, beam_force_vabs=FF, frame="material",
                                             s2_scheme=sch)["stress"])
    print("  %-12s  S13 %8.4f   S23 %8.4f    (in-plane S11 peak %8.2f)"
          % (sch, pk(S[:,4])/1e6, pk(S[:,3])/1e6, pk(S[:,0])/1e6))
print("\nthrough-thickness S13 profile (MPa)  z[mm]   VABS   mitc4_both")
Sb = np.asarray(dehom_rm.stress_at_points(B, coords, beam_force_vabs=FF, frame="material",
                                          s2_scheme="mitc4_both")["stress"])
for i in range(0, len(zmm), max(1, len(zmm)//12)):
    print("   %6.1f   %8.4f   %8.4f" % (zmm[i], V[i,4]/1e6, Sb[i,4]/1e6))
