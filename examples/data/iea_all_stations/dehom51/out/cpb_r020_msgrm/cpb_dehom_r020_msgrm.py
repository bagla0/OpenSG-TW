"""cpb_dehom_r020_msgrm.py -- regenerate the CPB-paper r/R=0.2 DEHOMOGENIZATION figures
from the MSG-RM production fields (xsec_r020_msgrm/data/msgrm_r020_fields.npz):

  (1) circumferential + spar-cap through-thickness LINE figures, CPB style:
      1x3 panels, VABS blue -o vs OpenSG-RM orange --s (open), non-dimensional path
      coordinate 0..1, stress in MPa, TOTAL local displacement in meters
      -> figures/dehom_r020_{circ,cap}_{stress,disp}.png
  (2) full-section contour ROW figures (VABS | OpenSG-RM + shared colorbar):
      Gauss-based exploded stress, nodal warping displacement
      -> figures/r020_row_{S11,S22,S12,u1,u2,u3}.png

The VABS side of the line plots is read from the archived path .out files
(out/dehom_vabs) so the sampling points are literally the published ones; the
MSG-RM values are looked up at those identical coordinates in the production npz.
Self-check: the npz VABS warping, pushed back through the beam kinematics, must
reproduce the .out total displacement.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))                 # dehom51
MSGRM = os.path.join(IEA, "out", "xsec_r020_msgrm")
VABS = os.path.join(IEA, "out", "VABS_iea51")
VBD = os.path.join(IEA, "out", "dehom_vabs")
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
VABSC = "#1f77b4"
RMC = "#ff7f0e"

z = np.load(os.path.join(MSGRM, "data", "msgrm_r020_fields.npz"))
sm_xy = z["sm_xy"]; sM = z["stress_msgrm"]; sV = z["stress_vabs"]     # MPa
xy = z["xy_nodes"]; uM_w = z["disp_msgrm"]; uV_w = z["disp_vabs"]     # warping, mm


def beam_kinematics(path, node):
    Lf = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(Lf):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in Lf[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")])
            RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]])
            t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
            return u_g, C
    raise ValueError("no BeamDyn header")


u_g, Cbk = beam_kinematics(os.path.join(VABS, "iea51vabs_bd_driver.out"), 11)


def total_disp(u_warp_mm, node_xy):
    """warping (mm, section frame) -> TOTAL local displacement (m), u = C(w+r)+u_g-r."""
    r3 = np.column_stack([np.zeros(len(node_xy)), node_xy[:, 0], node_xy[:, 1]])
    return (Cbk @ (u_warp_mm / 1e3 + r3).T).T + u_g - r3


def read_out(path):
    d = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


def plainaxis(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(alpha=0.3)


def rel_err(a, b):
    d = np.abs(a - b)
    keep = d <= 8.0 * np.median(d) + 1e-12
    return 100.0 * np.linalg.norm((a - b)[keep]) / (np.linalg.norm(b[keep]) + 1e-30)


tree_g = cKDTree(sm_xy)
tree_n = cKDTree(xy)
SCOMP = ["s_11", "s_12", "s_22"]
SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
SCOL = [0, 5, 1]                                   # npz VABS-order columns
UCOMP = ["u_1", "u_2", "u_3"]
ULAB = [r"$u_1$", r"$u_2$", r"$u_3$"]

PATHS = {"iea_s10.circumferential": "dehom_r020_circ",
         "iea_s10.lp_sparcap_left_thickness": "dehom_r020_cap"}
rep = ["=== CPB r/R=0.2 dehom line figures, MSG-RM fields ==="]
for stem, fout in PATHS.items():
    vb = read_out(os.path.join(VBD, stem + ".out"))
    P = np.column_stack([vb["y2"], vb["y3"]])
    s = vb["non_dim_path"]
    s = (s - s.min()) / (s.max() - s.min() + 1e-30)

    dg, ig = tree_g.query(P)
    dn, iN = tree_n.query(P)
    sMp = sM[ig]                                   # MPa, MSG-RM at the path gauss pts
    uMp = total_disp(uM_w[iN], xy[iN])             # m, total
    uVchk = total_disp(uV_w[iN], xy[iN])
    chk = max(abs(uVchk[:, k] - vb[UCOMP[k]]).max() for k in range(3))
    rep.append("%s: gauss-match max %.2e m, node-match max %.2e m, "
               "VABS total-disp reconstruction max diff %.2e m"
               % (stem, dg.max(), dn.max(), chk))

    d0 = np.abs(sMp[:, 0] - vb["s_11"] / 1e6)
    keep = d0 <= 8.0 * np.median(d0) + 1e-12
    ndrop = int((~keep).sum())

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    for k in range(3):
        ax = axs[k]
        ax.plot(s[keep], vb[SCOMP[k]][keep] / 1e6, "-o", color=VABSC, ms=3.5,
                lw=1.5, label="VABS")
        ax.plot(s[keep], sMp[keep, SCOL[k]], "--s", color=RMC, ms=3.5, mfc="none",
                mew=1.2, lw=1.4, label="OpenSG-RM")
        ax.set_ylabel("%s  [MPa]" % SLAB[k], fontsize=11)
        ax.set_xlabel("non-dimensional path coordinate", fontsize=10)
        plainaxis(ax)
        ax.legend(fontsize=9, loc="best")
    if ndrop:
        axs[0].text(0.03, 0.03, "%d ply-interface pt(s) hidden" % ndrop,
                    transform=axs[0].transAxes, va="bottom", fontsize=7, color="0.55")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fout + "_stress.png"), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for k in range(3):
        ax = axs[k]
        ax.plot(s, vb[UCOMP[k]], "-o", color=VABSC, ms=3.5, lw=1.5, label="VABS")
        ax.plot(s, uMp[:, k], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2, lw=1.4,
                label="OpenSG-RM")
        ax.set_ylabel("%s  [m]" % ULAB[k], fontsize=11)
        ax.set_xlabel("non-dimensional path coordinate", fontsize=10)
        plainaxis(ax)
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fout + "_disp.png"), dpi=150)
    plt.close(fig)

    rep.append("  stress: " + "  ".join("s%s %.1f%%" % (["11", "12", "22"][k],
               rel_err(sMp[:, SCOL[k]], vb[SCOMP[k]] / 1e6)) for k in range(3)))
    rep.append("  disp  : " + "  ".join("%s %.2f%%" % (["u1", "u2", "u3"][k],
               rel_err(uMp[:, k], vb[UCOMP[k]])) for k in range(3)))
    print("\n".join(rep[-3:]), flush=True)

# ---------------- contour rows ----------------
Lf = [l for l in open(os.path.join(VABS, "iea_s10.sg")).read().splitlines() if l.strip()]
hi = next(i for i, l in enumerate(Lf) if len(l.split()) == 3
          and all(x.lstrip("-").isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
nn, ne, nm = [int(x) for x in Lf[hi].split()]
conn = [[int(x) for x in Lf[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
trl = []
for c in conn:
    c0 = [n - 1 for n in c]
    if len(c0) == 3:
        trl.append(c0[:3])
    elif len(c0) >= 4:
        trl.extend([[c0[0], c0[1], c0[2]], [c0[0], c0[2], c0[3]]])
tris = np.array(trl)
M = tris.shape[0]
tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
gxy = sm_xy.reshape(M, 3, 2)
Ai = np.linalg.inv(np.concatenate([np.ones((M, 3, 1)), gxy], 2))
Cc = np.concatenate([np.ones((M, 3, 1)), xy[tris]], 2)
EP = xy[tris].reshape(-1, 2)
etri = mtri.Triangulation(EP[:, 0], EP[:, 1], np.arange(3 * M).reshape(M, 3))


def corners_of(val):
    g = val.reshape(M, 3, 1)
    c = Cc @ (Ai @ g)
    return np.clip(c, g.min(1, keepdims=True), g.max(1, keepdims=True)).reshape(-1)


def row_fig(name, exploded, dataV, dataR, label, unit):
    m = np.nanpercentile(np.abs(np.r_[dataV, dataR]), 99) or 1e-9
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 3.1))
    for a, dat, ttl in [(ax[0], dataV, "VABS"), (ax[1], dataR, "OpenSG-RM")]:
        tr = etri if exploded else tri
        cs = a.tripcolor(tr, np.clip(dat, -m, m), shading="gouraud", cmap="rainbow",
                         vmin=-m, vmax=m)
        a.set_aspect("equal"); a.axis("off")
        a.set_title(ttl, fontsize=15, pad=4)
    cb = fig.colorbar(cs, ax=ax.tolist(), shrink=0.92, pad=0.012)
    cb.set_label("%s %s" % (label, unit), fontsize=15)
    cb.ax.tick_params(labelsize=12)
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


for ci, lab, nmn in [(0, r"$\sigma_{11}$", "S11"), (1, r"$\sigma_{22}$", "S22"),
                     (5, r"$\sigma_{12}$", "S12")]:
    row_fig("r020_row_%s.png" % nmn, True,
            corners_of(sV[:, ci]), corners_of(sM[:, ci]), lab, "(MPa)")
for ci, lab, nmn in [(0, r"$u_1$", "u1"), (1, r"$u_2$", "u2"), (2, r"$u_3$", "u3")]:
    row_fig("r020_row_%s.png" % nmn, False, uV_w[:, ci], uM_w[:, ci], lab, "(mm)")

open(os.path.join(DATA, "dehom_report.txt"), "w").write("\n".join(rep) + "\n")
print("done", flush=True)
