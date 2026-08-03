"""cpb_r020_dualref_plot.py -- r=0.2 dehom line plots overlaying BOTH references:
VABS, OpenSG-RM (center / mid-surface), OpenSG-RM (OML), along the circumferential,
spar-cap through-thickness, and centre-web junction paths.  Reads the two field npz
(final_mid_fields.npz, final_oml_fields.npz) written by cpb_r020_final.py -- both are
evaluated at the SAME VABS .SM/.U coordinates, so the overlay is exact.

Outputs figures/dual_{circ,cap,junc}_{stress,disp}.png.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
VABS = os.path.join(IEA, "out", "VABS_iea51")
VBD = os.path.join(IEA, "out", "dehom_vabs")
BD_VABS = os.path.join(VABS, "iea51vabs_bd_driver.out")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

VABSC = "#1f77b4"
CENC = "#ff7f0e"     # center / mid-surface
OMLC = "#2ca02c"     # OML
IMLC = "#d62728"     # IML (present only if final_iml_fields.npz exists)

zc = np.load(os.path.join(DATA, "final_mid_fields.npz"))
zo = np.load(os.path.join(DATA, "final_oml_fields.npz"))
sm_xy = zc["sm_xy"]; xy = zc["xy_nodes"]
sVg = zc["stress_vabs"]; uV = zc["disp_vabs"]
sMc = zc["stress_msgrm"]; uMc = zc["disp_msgrm"]
sMo = zo["stress_msgrm"]; uMo = zo["disp_msgrm"]
_ip = os.path.join(DATA, "final_iml_fields.npz")
HAS_IML = os.path.exists(_ip)
if HAS_IML:
    zi = np.load(_ip)
    sMi = zi["stress_msgrm"]; uMi = zi["disp_msgrm"]
tree_g = cKDTree(sm_xy); tree_n = cKDTree(xy)


def beam_kin(path, node):
    Lf = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(Lf):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in Lf[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            return u_g, np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
    raise ValueError("no header")


u_g, Cbk = beam_kin(BD_VABS, 11)


def total(uw_mm, nod):
    r3 = np.column_stack([np.zeros(len(nod)), nod[:, 0], nod[:, 1]])
    return (Cbk @ (uw_mm / 1e3 + r3).T).T + u_g - r3


def read_out(p):
    d = {}
    for ln in open(p):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split(); d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


def plainaxis(ax):
    f = ScalarFormatter(useOffset=False); f.set_scientific(False)
    ax.yaxis.set_major_formatter(f); ax.grid(alpha=0.3)


SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
SCOL = [0, 5, 1]
SVB = ["s_11", "s_12", "s_22"]
ULAB = [r"$u_1$", r"$u_2$", r"$u_3$"]


def stress_panel(ax3, xdata, vb_vals, cvals, ovals, xlabel, keep=None, ivals=None):
    for k in range(3):
        ax = ax3[k]
        m = np.ones(len(xdata), bool) if keep is None else keep
        ax.plot(xdata[m], vb_vals[k][m], "-o", color=VABSC, ms=3.2, lw=1.5,
                label="VABS")
        ax.plot(xdata[m], cvals[k][m], "--s", color=CENC, ms=3.2, mfc="none", mew=1.1, lw=1.3,
                label="OpenSG-RM (center)")
        ax.plot(xdata[m], ovals[k][m], ":^", color=OMLC, ms=3.4, mfc="none", mew=1.1, lw=1.3,
                label="OpenSG-RM (OML)")
        if ivals is not None:
            ax.plot(xdata[m], ivals[k][m], "-.v", color=IMLC, ms=3.4, mfc="none", mew=1.1,
                    lw=1.1, label="OpenSG-RM (IML)")
        ax.set_ylabel("%s  [MPa]" % SLAB[k], fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10); plainaxis(ax)
        ax.legend(fontsize=8, loc="best")


def disp_panel(ax3, xdata, vb, cc, oo, xlabel, ii=None):
    for k in range(3):
        ax = ax3[k]
        ax.plot(xdata, vb[:, k], "-o", color=VABSC, ms=3.2, lw=1.5, label="VABS")
        ax.plot(xdata, cc[:, k], "--s", color=CENC, ms=3.2, mfc="none", mew=1.1, lw=1.3,
                label="OpenSG-RM (center)")
        ax.plot(xdata, oo[:, k], ":^", color=OMLC, ms=3.4, mfc="none", mew=1.1, lw=1.3,
                label="OpenSG-RM (OML)")
        if ii is not None:
            ax.plot(xdata, ii[:, k], "-.v", color=IMLC, ms=3.4, mfc="none", mew=1.1, lw=1.1,
                    label="OpenSG-RM (IML)")
        ax.set_ylabel("%s  [m]" % ULAB[k], fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10); plainaxis(ax)
        ax.legend(fontsize=8, loc="best")


# ---------- circumferential + cap paths (from the VABS .out) ----------
for stem, tag, xlab in (("iea_s10.circumferential", "circ", "non-dimensional path coordinate"),
                        ("iea_s10.lp_sparcap_left_thickness", "cap", "non-dimensional path coordinate")):
    vb = read_out(os.path.join(VBD, stem + ".out"))
    Pp = np.column_stack([vb["y2"], vb["y3"]])
    s = vb["non_dim_path"]; s = (s - s.min()) / (s.max() - s.min() + 1e-30)
    ig = tree_g.query(Pp)[1]; iN = tree_n.query(Pp)[1]
    dc = np.abs(sMc[ig, 0] - vb["s_11"] / 1e6); do = np.abs(sMo[ig, 0] - vb["s_11"] / 1e6)
    keep = (dc <= 8.0 * np.median(dc) + 1e-12) & (do <= 8.0 * np.median(do) + 1e-12)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    stress_panel(axs, s, [vb[c] / 1e6 for c in SVB],
                 [sMc[ig, SCOL[k]] for k in range(3)], [sMo[ig, SCOL[k]] for k in range(3)],
                 xlab, keep, ivals=[sMi[ig, SCOL[k]] for k in range(3)] if HAS_IML else None)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "dual_%s_stress.png" % tag), dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    disp_panel(axs, s, total(uV[iN], xy[iN]), total(uMc[iN], xy[iN]), total(uMo[iN], xy[iN]), xlab,
               ii=total(uMi[iN], xy[iN]) if HAS_IML else None)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "dual_%s_disp.png" % tag), dpi=150); plt.close(fig)
    print("wrote dual_%s_{stress,disp}.png" % tag, flush=True)

# ---------- junction polyline ----------
jv = np.loadtxt(os.path.join(DATA, "junction_polyline_oml.dat"))
p0, p1, p2 = jv[0], jv[1], jv[2]
s1 = np.arange(0.0, np.linalg.norm(p1 - p0), 0.0025)
s2 = np.arange(0.0, np.linalg.norm(p2 - p1), 0.0025)
n1 = (p1 - p0) / np.linalg.norm(p1 - p0); n2 = (p2 - p1) / np.linalg.norm(p2 - p1)
pline = np.vstack([p0[None] + s1[:, None] * n1[None], p1[None] + s2[:, None] * n2[None]])
arc = np.concatenate([s1, np.linalg.norm(p1 - p0) + s2]) * 1e3
hj_mm = np.linalg.norm(p1 - p0) * 1e3


def dedup(tree, maxd):
    dd, ii = tree.query(pline); ok = dd < maxd
    seen, aa, jj = set(), [], []
    for a_, j_ in zip(arc[ok], ii[ok]):
        if j_ not in seen:
            seen.add(j_); aa.append(a_); jj.append(j_)
    return np.array(aa), np.array(jj, int)


aj, gj = dedup(tree_g, 0.012); an, nj = dedup(tree_n, 0.012)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
stress_panel(axs, aj, [sVg[gj, SCOL[k]] for k in range(3)],
             [sMc[gj, SCOL[k]] for k in range(3)], [sMo[gj, SCOL[k]] for k in range(3)],
             "arc along the junction path [mm]",
             ivals=[sMi[gj, SCOL[k]] for k in range(3)] if HAS_IML else None)
for ax in axs:
    ax.axvline(hj_mm, color="0.6", lw=1.0, ls=":")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "dual_junc_stress.png"), dpi=150); plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
disp_panel(axs, an, total(uV[nj], xy[nj]), total(uMc[nj], xy[nj]), total(uMo[nj], xy[nj]),
           "arc along the junction path [mm]", ii=total(uMi[nj], xy[nj]) if HAS_IML else None)
for ax in axs:
    ax.axvline(hj_mm, color="0.6", lw=1.0, ls=":")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "dual_junc_disp.png"), dpi=150); plt.close(fig)
print("wrote dual_junc_{stress,disp}.png", flush=True)
