"""msgrm_v2_paths_transverse.py -- THROUGH-THICKNESS stress components (s13, s23, s33)
along the three r=0.2 paths: VABS vs first-order vs V2 recovery, CPB sampling
(VABS path-extraction .out files for circ/cap; material-aware projection for the
junction polyline).

Expected from theory (checked here): V2 changes s13/s23 by EXACTLY zero (monoclinic
parity -- their second-order correction lives in V3), and adds only the tiny
self-equilibrated s33 (dE11 = 0 for the constant-shear beam load; the surface-pressure
part would need V2L, and the VABS reference run itself carries no surface pressure).
Frame note: recovered components are wall-frame; on the cap/circ (near-horizontal
walls) they compare directly with the section-frame VABS values, on the web leg of the
junction polyline the transverse components mix with in-plane ones (flagged).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import msgrm_v2_junction as J
import msgrm_v2_paths as PATHS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); DATA = os.path.join(HERE, "data")
VBD = os.path.join(J.IEA, "out", "dehom_vabs")
CPB = os.path.join(HERE, "..", "cpb_r020_msgrm", "data", "final_mid_fields.npz")

TCOMP = ["s_13", "s_23", "s_33"]
TLAB = [r"$\sigma_{13}$", r"$\sigma_{23}$", r"$\sigma_{33}$"]
TCOL = [4, 3, 2]
VABSC = "k"; RM1C = "#1f77b4"; RM2C = "#ff7f0e"


def read_out(path):
    d = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


def recover_at(points, second, el=None, zd=None):
    if el is None:
        el, xi, zd_ = J.fast_project(points)
        zd = zd_
    else:
        P = np.atleast_2d(points)
        xi = np.clip(((P - J.A0[el]) * J.T[el]).sum(1) / J.L2[el], 0.0, 1.0)
    out = np.zeros((len(el), 6))
    for k in range(len(el)):
        out[k] = J.recover(int(el[k]), float(xi[k]), float(zd[k]) - J.zoff[int(el[k])],
                           second) / 1e6
    return out


rep = ["=== through-thickness components: VABS vs first-order vs V2 (CPB sampling) ==="]
figpaths = []

# ---------- circ + cap from the VABS path-extraction files ----------
for stem, pth, xlab in (("iea_s10.circumferential", "circ", "non-dimensional path coordinate"),
                        ("iea_s10.lp_sparcap_left_thickness", "cap", "non-dimensional path (OML -> IML)")):
    vb = read_out(os.path.join(VBD, stem + ".out"))
    Pp = np.column_stack([vb["y2"], vb["y3"]])
    s = vb["non_dim_path"]
    s = (s - s.min()) / (s.max() - s.min() + 1e-30)
    s1 = recover_at(Pp, False)
    s2 = recover_at(Pp, True)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    for kk in range(3):
        ax = axs[kk]
        ax.plot(s, vb[TCOMP[kk]] / 1e6, "-o", color=VABSC, ms=3.5, lw=1.4, label="VABS")
        ax.plot(s, s1[:, TCOL[kk]], "--s", color=RM1C, ms=3.5, mfc="none", mew=1.2, lw=1.3,
                label="MSG-RM 1st order")
        ax.plot(s, s2[:, TCOL[kk]], ":^", color=RM2C, ms=3.5, mfc="none", mew=1.2, lw=1.3,
                label="MSG-RM V2")
        ax.set_ylabel("%s  [MPa]" % TLAB[kk], fontsize=12)
        ax.set_xlabel(xlab, fontsize=11)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=10, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG, "r020_transverse_v2_%s.png" % pth)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    figpaths.append(out)

    line = "[%s] %d pts | rms diff (MPa) first->V2 : " % (pth, len(s))
    line += "  ".join("%s %.4f->%.4f" % (TCOMP[kk],
                      float(np.sqrt(np.mean((s1[:, TCOL[kk]] - vb[TCOMP[kk]] / 1e6) ** 2))),
                      float(np.sqrt(np.mean((s2[:, TCOL[kk]] - vb[TCOMP[kk]] / 1e6) ** 2))))
                      for kk in range(3))
    line += " | VABS rms: " + "  ".join("%.4f" % float(np.sqrt(np.mean((vb[TCOMP[kk]] / 1e6) ** 2)))
                                        for kk in range(3))
    rep.append(line); print(line, flush=True)

# ---------- junction polyline, material-aware ----------
cpb = np.load(CPB)
el_ma = cpb["el_gauss"]; zd_ma = cpb["z_gauss"]; zoff_ma = cpb["zoff"]
aj, gj, hj_mm = PATHS.junction_polyline()
o = np.argsort(aj); aj, gj = aj[o], gj[o]
elj = el_ma[gj].astype(int)
zj = zd_ma[gj] - zoff_ma[elj] + J.zoff[elj]        # recover_at subtracts J.zoff again
s1 = recover_at(J.sm_xy[gj], False, el=elj, zd=zj)
s2 = recover_at(J.sm_xy[gj], True, el=elj, zd=zj)
sV = J.sVg[gj]

fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
for kk in range(3):
    ax = axs[kk]
    ax.plot(aj, sV[:, TCOL[kk]], "-o", color=VABSC, ms=3.5, lw=1.4, label="VABS")
    ax.plot(aj, s1[:, TCOL[kk]], "--s", color=RM1C, ms=3.5, mfc="none", mew=1.2, lw=1.3,
            label="MSG-RM 1st order")
    ax.plot(aj, s2[:, TCOL[kk]], ":^", color=RM2C, ms=3.5, mfc="none", mew=1.2, lw=1.3,
            label="MSG-RM V2")
    ax.set_ylabel("%s  [MPa]" % TLAB[kk], fontsize=12)
    ax.set_xlabel("polyline position [mm]  (cap wall then web)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.axvline(hj_mm, color="0.6", lw=1.0, ls=":")
axs[0].legend(fontsize=10, frameon=False)
fig.tight_layout()
out = os.path.join(FIG, "r020_transverse_v2_junc.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
figpaths.append(out)

line = "[junc] %d pts (material-aware) | rms diff (MPa) first->V2 : " % len(gj)
line += "  ".join("%s %.4f->%.4f" % (TCOMP[kk],
                  float(np.sqrt(np.mean((s1[:, TCOL[kk]] - sV[:, TCOL[kk]]) ** 2))),
                  float(np.sqrt(np.mean((s2[:, TCOL[kk]] - sV[:, TCOL[kk]]) ** 2))))
                  for kk in range(3))
line += " | VABS rms: " + "  ".join("%.4f" % float(np.sqrt(np.mean(sV[:, TCOL[kk]] ** 2)))
                                    for kk in range(3))
line += "   [web leg: wall frame != section frame for transverse comps]"
rep.append(line); print(line, flush=True)
print("max |V2 - first| per comp: s13 %.2e  s23 %.2e  s33 %.2e MPa (parity: s13/s23 -> 0)"
      % tuple(float(np.max(np.abs(s2[:, c] - s1[:, c]))) for c in TCOL), flush=True)

open(os.path.join(DATA, "README_v2_transverse_paths.txt"), "w").write("\n".join(rep) + "\n")
print("figures: r020_transverse_v2_{circ,cap,junc}.png", flush=True)
