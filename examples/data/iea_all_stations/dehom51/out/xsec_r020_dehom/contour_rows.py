"""contour_rows.py -- the r=0.2 contour comparison re-laid ROW-WISE: one PNG per
field row (VABS | OpenSG-RM side by side + colorbar), center-ref, from the cached
RM evaluations of r020_dehom_s10.py.  Same conformal convention: Gauss-based
exploded stress, nodal displacement.  Outputs figures/r020_row_{S11,S22,S12,
u1,u2,u3}.png"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

HERE = os.path.dirname(os.path.abspath(__file__))
D51 = os.path.abspath(os.path.join(HERE, "..", ".."))     # .../dehom51
VABS = os.path.join(D51, "out", "VABS_iea51")
FIG = os.path.join(HERE, "figures")
CMAP = "rainbow"


def beam_kinematics(path, node):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]]); return u_g, C
    raise ValueError("no BeamDyn header")


def parse_sg_conn(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    hi = next(i for i, l in enumerate(L) if len(l.split()) == 3
              and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
    nn, ne, nm = [int(x) for x in L[hi].split()]
    conn = [[int(x) for x in L[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
    return nn, conn


U = np.loadtxt(os.path.join(VABS, "iea_s10.sg.U")); U = U[np.argsort(U[:, 0])]
xy = U[:, 1:3]; uV_tot = U[:, 3:6]
nn, conn = parse_sg_conn(os.path.join(VABS, "iea_s10.sg"))
trl = []
for c in conn:
    c0 = [n - 1 for n in c]
    trl.append(c0[:3]) if len(c0) == 3 else (trl.extend([[c0[0], c0[1], c0[2]], [c0[0], c0[2], c0[3]]]) if len(c0) == 4 else None)
tris = np.array(trl); tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris); M = tris.shape[0]

u_g, C = beam_kinematics(os.path.join(VABS, "iea51vabs_bd_driver.out"), 11)
Cinv = np.linalg.inv(C)
r3 = np.column_stack([np.zeros(len(xy)), xy[:, 0], xy[:, 1]])
uV = ((Cinv @ (uV_tot - u_g + r3).T).T - r3) * 1e3

dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]; sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6

z = np.load(os.path.join(HERE, "_rm_s10_cache.npz"))
uR = z["uR"]; sRg = z["sRg"]

gxy = sm_xy.reshape(M, 3, 2)
Ai = np.linalg.inv(np.concatenate([np.ones((M, 3, 1)), gxy], 2))
Cc = np.concatenate([np.ones((M, 3, 1)), xy[tris]], 2)
cornerV = Cc @ (Ai @ sVg.reshape(M, 3, 6)); cornerR = Cc @ (Ai @ sRg.reshape(M, 3, 6))
gV6 = sVg.reshape(M, 3, 6); gR6 = sRg.reshape(M, 3, 6)
cornerV = np.clip(cornerV, gV6.min(1, keepdims=True), gV6.max(1, keepdims=True))
cornerR = np.clip(cornerR, gR6.min(1, keepdims=True), gR6.max(1, keepdims=True))
EP = xy[tris].reshape(-1, 2)
etri = mtri.Triangulation(EP[:, 0], EP[:, 1], np.arange(3 * M).reshape(M, 3))
sVe = cornerV.reshape(-1, 6); sRe = cornerR.reshape(-1, 6)
print("prep done: %d tris" % M)


def row_fig(name, exploded, dataV, dataR, label, unit):
    m = np.nanpercentile(np.abs(np.r_[dataV, dataR]), 99) or 1e-9
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 3.1))
    for a, dat, ttl in [(ax[0], dataV, "VABS"), (ax[1], dataR, "OpenSG-RM")]:
        tr = etri if exploded else tri
        cs = a.tripcolor(tr, np.clip(dat, -m, m), shading="gouraud", cmap=CMAP,
                         vmin=-m, vmax=m)
        a.set_aspect("equal"); a.axis("off")
        a.set_title(ttl, fontsize=15, pad=4)
    cb = fig.colorbar(cs, ax=ax.tolist(), shrink=0.92, pad=0.012)
    cb.set_label("%s %s" % (label, unit), fontsize=15)
    cb.ax.tick_params(labelsize=12)
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


for ci, lab, nm in [(0, r"$\sigma_{11}$", "S11"), (1, r"$\sigma_{22}$", "S22"),
                    (5, r"$\sigma_{12}$", "S12")]:
    row_fig("r020_row_%s.png" % nm, True, sVe[:, ci], sRe[:, ci], lab, "(MPa)")
for ci, lab, nm in [(0, r"$u_1$", "u1"), (1, r"$u_2$", "u2"), (2, r"$u_3$", "u3")]:
    row_fig("r020_row_%s.png" % nm, False, uV[:, ci], uR[:, ci], lab, "(mm)")
