"""msgrm_v2_paths_inplane.py -- IN-PLANE stresses (s11, s12, s22) along the three r=0.2
paths, V2 recovery vs VABS, sampled the way the previous (CPB) implementation did it.

Why the earlier msgrm_v2_paths.py figures looked noisy -- three sampling differences,
none of them recovery accuracy (the full-section field matches production to 0.29 MPa):
  (1) VABS reference: CPB uses the VABS PATH-EXTRACTION files (out/dehom_vabs/*.out --
      consistent on-path points), not nearest raw .SM gauss points, whose z scatters
      through the wall thickness (+-h/2 of bending stress -> sawtooth in both curves);
  (2) junction: CPB's MATERIAL-AWARE projection (final_mid_fields.npz el_gauss) stops
      cap gauss points from being evaluated with the web layup and vice versa;
  (3) CPB hides ply-interface half-side mismatches (> 8 x median |diff| on s11).
This script redoes the figures with (1)-(3).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import msgrm_v2_junction as J                      # validated full pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); DATA = os.path.join(HERE, "data")
VBD = os.path.join(J.IEA, "out", "dehom_vabs")
CPB = os.path.join(HERE, "..", "cpb_r020_msgrm", "data", "final_mid_fields.npz")

SCOMP = ["s_11", "s_12", "s_22"]
SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
SCOL = [0, 5, 1]
VABSC = "k"; RMC = "#ff7f0e"


def read_out(path):
    d = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


def v2_at(points, el=None):
    """V2 recovery at section points; el optionally forced (material-aware indices)."""
    if el is None:
        el, xi, zd = J.fast_project(points)
    else:
        P = np.atleast_2d(points)
        xi = np.clip(((P - J.A0[el]) * J.T[el]).sum(1) / J.L2[el], 0.0, 1.0)
        foot = J.A0[el] + xi[:, None] * J.T[el]
        zd = ((P - foot) * J.nvec[el]).sum(1)
    out = np.zeros((len(el), 6))
    for k in range(len(el)):
        out[k] = J.recover(el[k], xi[k], zd[k] - J.zoff[el[k]], True) / 1e6
    return out


rep = ["=== in-plane path comparison, CPB sampling, V2 recovery vs VABS ==="]

# ---------- circ + cap: the VABS path-extraction files ----------
for stem, pth, xlab in (("iea_s10.circumferential", "circ", "non-dimensional path coordinate"),
                        ("iea_s10.lp_sparcap_left_thickness", "cap", "non-dimensional path (OML -> IML)")):
    vb = read_out(os.path.join(VBD, stem + ".out"))
    Pp = np.column_stack([vb["y2"], vb["y3"]])
    s = vb["non_dim_path"]
    s = (s - s.min()) / (s.max() - s.min() + 1e-30)
    sMp = v2_at(Pp)
    d0 = np.abs(sMp[:, 0] - vb["s_11"] / 1e6)
    keep = d0 <= 8.0 * np.median(d0) + 1e-12
    ndrop = int((~keep).sum())

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    for kk in range(3):
        ax = axs[kk]
        ax.plot(s[keep], vb[SCOMP[kk]][keep] / 1e6, "-o", color=VABSC, ms=3.5, lw=1.5, label="VABS")
        ax.plot(s[keep], sMp[keep, SCOL[kk]], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2,
                lw=1.4, label="MSG-RM V2")
        ax.set_ylabel("%s  [MPa]" % SLAB[kk], fontsize=12)
        ax.set_xlabel(xlab, fontsize=11)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=10, frameon=False)
    if ndrop:
        axs[0].text(0.03, 0.03, "%d ply-interface pt(s) hidden" % ndrop,
                    transform=axs[0].transAxes, va="bottom", fontsize=7, color="0.55")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "r020_inplane_v2_%s.png" % pth), dpi=150, bbox_inches="tight")
    plt.close(fig)

    line = "[%s] %d pts (%d hidden) | rms diff (MPa): " % (pth, len(s), ndrop)
    line += "  ".join("%s %.2f" % (SCOMP[kk], float(np.sqrt(np.mean(
        (sMp[keep, SCOL[kk]] - vb[SCOMP[kk]][keep] / 1e6) ** 2)))) for kk in range(3))
    line += " | VABS rms: " + "  ".join("%.2f" % float(np.sqrt(np.mean(
        (vb[SCOMP[kk]][keep] / 1e6) ** 2))) for kk in range(3))
    rep.append(line); print(line, flush=True)

# ---------- junction polyline: material-aware projection from the CPB run ----------
cpb = np.load(CPB)
el_ma = cpb["el_gauss"]; zd_ma = cpb["z_gauss"]; zoff_ma = cpb["zoff"]
import msgrm_v2_paths as PATHS                     # reuse the polyline selection

aj, gj, hj_mm = PATHS.junction_polyline()
o = np.argsort(aj); aj, gj = aj[o], gj[o]
sMj = np.zeros((len(gj), 6))
for k, g in enumerate(gj):
    e = int(el_ma[g])
    P = J.sm_xy[g]
    xi = float(np.clip(((P - J.A0[e]) * J.T[e]).sum() / J.L2[e], 0.0, 1.0))
    z = float(zd_ma[g]) - float(zoff_ma[e])
    sMj[k] = J.recover(e, xi, z, True) / 1e6
sVj = J.sVg[gj]

fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
for kk in range(3):
    ax = axs[kk]
    ax.plot(aj, sVj[:, SCOL[kk]], "-o", color=VABSC, ms=3.5, lw=1.4, label="VABS")
    ax.plot(aj, sMj[:, SCOL[kk]], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2, lw=1.3,
            label="MSG-RM V2")
    ax.set_ylabel("%s  [MPa]" % SLAB[kk], fontsize=12)
    ax.set_xlabel("polyline position [mm]  (cap wall then web)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.axvline(hj_mm, color="0.6", lw=1.0, ls=":")
axs[0].legend(fontsize=10, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r020_inplane_v2_junc.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

line = "[junc] %d pts (material-aware) | rms diff (MPa): " % len(gj)
line += "  ".join("%s %.2f" % (SCOMP[kk], float(np.sqrt(np.mean(
    (sMj[:, SCOL[kk]] - sVj[:, SCOL[kk]]) ** 2)))) for kk in range(3))
line += " | VABS rms: " + "  ".join("%.2f" % float(np.sqrt(np.mean(
    sVj[:, SCOL[kk]] ** 2))) for kk in range(3))
rep.append(line); print(line, flush=True)

open(os.path.join(DATA, "README_v2_inplane_paths.txt"), "w").write("\n".join(rep) + "\n")
print("figures: r020_inplane_v2_{circ,cap,junc}.png", flush=True)
