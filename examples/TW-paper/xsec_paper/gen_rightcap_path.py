"""Build the LP RIGHT spar-cap through-thickness path FROM the VABS gauss points at a clean
right-of-centre cap column (a full laminate: glass / carbon / glass with both interfaces).  Every
path point IS a VABS gauss point, so the comparison is exact in both -- no interpolation, no
lateral penalty.  Writes solid.lp_sparcap_right_thickness_r020.coords, then compares plain-nearest
VABS against the RM two-step recovery at those exact coordinates."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm

D2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "data", "2d_yaml"))
SHELL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "TW-paper",
                                     "iea22_blade", "data", "shell_r020.yaml"))
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def row(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


ds = yaml.safe_load(open(os.path.join(D2, "iea_r020_solid.yaml")))
nodes = np.array([row(n)[:2] for n in ds["nodes"]])
tris = [[int(round(x)) - 1 for x in row(e)] for e in ds["elements"]]
mat = np.zeros(len(tris), int)
for grp in ds["sets"]["element"]:
    mi = int("".join(c for c in grp["name"] if c.isdigit()))
    for lab in grp["labels"]:
        mat[int(lab) - 1] = mi
cen = nodes.mean(0)
ecen = np.array([nodes[t].mean(0) for t in tris])
cap = ecen[(mat == 4) & (ecen[:, 1] > cen[1])]                     # carbon LP cap

d = np.loadtxt(os.path.join(D2, "IEA_VABS/iea_r020.sg.SM"), skiprows=2)
sm_xy, sm_s = d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]            # [S11,S22,S33,S23,S13,S12]

xc = float(np.percentile(cap[:, 0], 65))                           # right of centre, full clean stack
band = 6e-4
m = (np.abs(sm_xy[:, 0] - xc) < band) & (sm_xy[:, 1] > cap[:, 1].min() - 0.012)
col = sm_xy[m]
col = col[np.argsort(-col[:, 1])]                                  # OML (high y3) -> IML
# collapse near-duplicate depths (keep the one closest to the column axis xc)
keep, last = [], None
for i in range(len(col)):
    if last is None or abs(col[i, 1] - last) > 2e-4:
        keep.append(i); last = col[i, 1]
    elif abs(col[i, 0] - xc) < abs(col[keep[-1], 0] - xc):
        keep[-1] = i
col = col[keep]
print("RIGHT cap column xc=%.4f  %d exact gauss points, depth %.1f mm" %
      (xc, len(col), 1e3 * (col[0, 1] - col[-1, 1])))

# write coords (these ARE gauss points)
out = os.path.join(D2, "solid.lp_sparcap_right_thickness_r020.coords")
z = np.r_[0.0, np.cumsum(np.hypot(np.diff(col[:, 0]), np.diff(col[:, 1])))]
with open(out, "w") as f:
    for (x, y), a in zip(col, 100.0 * z / z[-1]):
        f.write("%.11f %.11f %.11f\n" % (x, y, a))
print("wrote", os.path.basename(out))

# ---- exact-point check + plain comparison (no penalty) ----
tree = cKDTree(sm_xy)
dist, idx = tree.query(col)
print("max VABS gauss match distance: %.2e m (all path points ARE gauss points)" % dist.max())
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")
Srm = np.asarray(dehom_rm.stress_at_points(B, col, beam_force_vabs=FF, frame="material",
                                           n_per_layer=4)["stress"]) / 1e6
V = sm_s[idx] / 1e6
print("\ndep(mm)  VABS S11   RM S11 |  VABS S12   RM S12")
for i in range(len(col)):
    print("  %5.1f   %8.2f  %8.2f | %7.3f  %7.3f"
          % (1e3 * (col[0, 1] - col[i, 1]), V[i, 0], Srm[i, 0], V[i, 5], Srm[i, 5]))
ip = np.linalg.norm(Srm[:, [0, 1, 5]] - V[:, [0, 1, 5]]) / np.linalg.norm(V[:, [0, 1, 5]]) * 100
print("in-plane ||.|| = %.1f%%  (plain nearest, no penalty)" % ip)
