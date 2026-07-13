"""Debug the LP spar-cap through-thickness sampling: why the 2nd node (outer interface) picks
an anomalous VABS gauss-point stress.  Prints per path point: coord, cumulative depth, the VABS
nearest-gauss position + distance + stress, and the RM-recovered stress + mapped depth/ply."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm

D2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "data", "2d_yaml"))
VABS = os.path.join(D2, "IEA_VABS")
SHELL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "TW-paper",
                                     "iea22_blade", "data", "shell_r020.yaml"))
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm(path):
    d = np.loadtxt(path, skiprows=2)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]     # xy, [S11,S22,S33,S23,S13,S12]


cap = np.loadtxt(os.path.join(D2, "solid.lp_sparcap_left_thickness_r020.coords"))[:, :2]
sm_xy, sm_s = load_sm(os.path.join(VABS, "iea_r020.sg.SM"))
tree = cKDTree(sm_xy)
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")
res = dehom_rm.stress_at_points(B, cap, beam_force_vabs=FF, frame="material")
Srm = np.asarray(res["stress"]) / 1e6
dep_rm = np.asarray(res["depth"])

z = np.r_[0.0, np.cumsum(np.hypot(np.diff(cap[:, 0]), np.diff(cap[:, 1])))] * 1e3   # mm
dist, idx = tree.query(cap)
# also the k=5 nearest to see the local spread
kd, ki = tree.query(cap, k=6)

print("cap column: %d points, span y2=[%.4f,%.4f] y3=[%.4f,%.4f]" %
      (len(cap), cap[:, 0].min(), cap[:, 0].max(), cap[:, 1].min(), cap[:, 1].max()))
print("%3s %8s %8s %7s | %8s %8s %8s | %7s %7s %7s | %8s %8s %8s | RMdepth" %
      ("i", "y2", "y3", "depth", "gy2", "gy3", "gdist", "V.S11", "V.S22", "V.S12",
       "R.S11", "R.S22", "R.S12"))
for i in range(min(len(cap), 8)):
    gv = sm_s[idx[i]] / 1e6
    # spread of S12 over the 6 nearest gauss pts (to see if the nearest is an outlier)
    s12_k = sm_s[ki[i], 5] / 1e6
    print("%3d %8.4f %8.4f %7.2f | %8.4f %8.4f %7.4f | %7.2f %7.3f %7.3f | %8.2f %8.3f %8.3f | %6.2f"
          % (i, cap[i, 0], cap[i, 1], z[i], sm_xy[idx[i], 0], sm_xy[idx[i], 1], dist[i],
             gv[0], gv[1], gv[5], Srm[i, 0], Srm[i, 1], Srm[i, 5], dep_rm[i] * 1e3))
    print("      k=6 nearest gauss S12 (MPa):", np.array2string(s12_k, precision=3, suppress_small=True),
          " dists:", np.array2string(kd[i], precision=4))
print("\n... last 3 points (near IML) ...")
for i in range(len(cap) - 3, len(cap)):
    gv = sm_s[idx[i]] / 1e6
    print("%3d depth=%.2f | gdist=%.4f | V.S11=%.2f V.S22=%.3f V.S12=%.3f | R.S11=%.2f R.S22=%.3f R.S12=%.3f"
          % (i, z[i], dist[i], gv[0], gv[1], gv[5], Srm[i, 0], Srm[i, 1], Srm[i, 5]))
