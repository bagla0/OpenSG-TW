"""msgrm_v2_paths.py -- ALL six stress components along the three r=0.2 paths,
V2 (Eq. 66) recovery vs VABS .SM:

  circ : iea_s10.circumferential.coords            (mid-wall, LE around the section)
  cap  : iea_s10.lp_sparcap_left_thickness.coords  (spar-cap through-thickness, OML->IML)
  junc : CONNECTED polyline at the centre-web / spar-cap T-junction (cap wall OML-normal,
         then down the web mid-line -- the cpb_r020_final.py construction)

Reuses the validated full-section fields of msgrm_v2_junction.py (first order == the
production pipeline to 0.29 MPa) by importing it; here only path selection + figures.
Comparison frame: section frame at the .SM gauss points (the production convention);
on the web leg of the junction path the wall frame differs from the section frame --
in-plane components remain comparable, transverse ones mix (noted in the README).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import msgrm_v2_junction as J                      # runs the full pipeline (validated)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); DATA = os.path.join(HERE, "data")

sVg, sM1, sM2, sm_xy = J.sVg, J.sM1, J.sM2, J.sm_xy
corners, rc, deg, nd_el = J.corners, J.rc, J.deg, J.nd_el
is_web, nvec, layups, ldb = J.is_web, J.nvec, J.layups, J.ldb
frac = J.frac
tree_g = cKDTree(sm_xy)

NAMES = [r"$\sigma_{11}$", r"$\sigma_{22}$", r"$\sigma_{33}$",
         r"$\sigma_{23}$", r"$\sigma_{13}$", r"$\sigma_{12}$"]
ORDER = [0, 1, 5, 2, 4, 3]                          # panel order: 11, 22, 12 / 33, 13, 23
VABSC = "k"; V2C = "#ff7f0e"


def nearest_gauss(pts, maxd=0.012):
    d_, i_ = tree_g.query(np.atleast_2d(pts))
    ok = d_ < maxd
    return i_[ok], d_[ok], np.where(ok)[0]


def path_from_coords(fname):
    c = np.loadtxt(os.path.join(J.IEA, "coords", fname))
    gi, dist, keep = nearest_gauss(c[:, :2])
    return c[keep, 2], gi, dist.max()


def junction_polyline():
    """cpb_r020_final.py construction: OML-normal through the cap wall at the centre-web
    T-junction, then 1.6*h down the web mid-line; gauss pts within one cell of the line."""
    jn = [nd for nd in np.where(deg >= 3)[0] if corners[nd, 1] > 0]
    ndj = jn[int(np.argmin([abs(corners[nd, 0] + 0.044) for nd in jn]))]
    sk_e = [e for e in nd_el[ndj] if not is_web[e]]
    wb_e = [e for e in nd_el[ndj] if is_web[e]]
    nrm = np.mean([nvec[e] for e in sk_e], axis=0); nrm /= np.linalg.norm(nrm)
    e_w = wb_e[0]
    oth = int(rc[e_w, 1]) if int(rc[e_w, 0]) == ndj else int(rc[e_w, 0])
    dweb = corners[oth] - corners[ndj]; dweb /= np.linalg.norm(dweb)
    hj = float(sum(ldb[layups[sk_e[0]]]["thick"]))
    p0 = corners[ndj] - frac * hj * nrm
    p1 = p0 + hj * nrm
    s1 = np.arange(0.0, hj, 0.0025)
    s2 = np.arange(0.0, 1.6 * hj, 0.0025)
    pline = np.vstack([p0[None] + s1[:, None] * nrm[None],
                       p1[None] + s2[:, None] * dweb[None]])
    arc = np.concatenate([s1, hj + s2]) * 1e3
    d_, i_ = tree_g.query(pline)
    ok = d_ < 0.012
    seen, aa, ii = set(), [], []
    for a_, j_ in zip(arc[ok], i_[ok]):
        if j_ not in seen:
            seen.add(j_); aa.append(a_); ii.append(j_)
    return np.array(aa), np.array(ii, int), 1e3 * hj


paths = {}
paths["circ"] = ("arc position [mm]", *path_from_coords("iea_s10.circumferential.coords")[:2], None)
paths["cap"] = ("thickness position [mm]", *path_from_coords("iea_s10.lp_sparcap_left_thickness.coords")[:2], None)
aj, gj, hj_mm = junction_polyline()
paths["junc"] = ("polyline position [mm]  (cap wall then web)", aj, gj, hj_mm)

rep = ["=== r=0.2 all-component path comparison: V2 (Eq. 66) vs VABS ==="]
for tag, (xlab, arc, gi, divider) in paths.items():
    o = np.argsort(arc)
    arc, gi = np.asarray(arc)[o], np.asarray(gi)[o]
    sV = sVg[gi]; s2_ = sM2[gi]; s1_ = sM1[gi]
    fig, axs = plt.subplots(2, 3, figsize=(15.0, 8.2))
    for k, ci in enumerate(ORDER):
        ax = axs.flat[k]
        ax.plot(arc, sV[:, ci], "-o", color=VABSC, ms=3, lw=1.3, label="VABS")
        ax.plot(arc, s2_[:, ci], "-s", color=V2C, ms=3, lw=1.3, mfc="none",
                label="MSG-RM V2 (Eq. 66)")
        ax.set_xlabel(xlab); ax.set_ylabel(NAMES[ci] + " [MPa]")
        ax.grid(alpha=0.3)
        if divider is not None:
            ax.axvline(divider, color="0.6", lw=1.0, ls=":")
    axs.flat[0].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG, "r020_paths_v2_%s.png" % tag)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    line = "[%s] %d samples | rms diff vs VABS (MPa): " % (tag, len(gi))
    line += "  ".join("%s %.3f" % (n.strip("$\\sigma_{}"), float(np.sqrt(np.mean((sV[:, c] - s2_[:, c]) ** 2))))
                      for n, c in zip(np.array(NAMES)[ORDER], ORDER))
    rep.append(line)
    rep.append("[%s]      VABS signal rms       : " % tag + "  ".join(
        "%s %.3f" % (np.array(NAMES)[c].strip("$\\sigma_{}"), float(np.sqrt(np.mean(sV[:, c] ** 2))))
        for c in ORDER))
    print(rep[-2]); print(rep[-1], flush=True)

np.savez(os.path.join(DATA, "msgrm_v2_paths.npz"),
         **{"%s_%s" % (t, k): v for t, (xl, a_, g_, dv) in paths.items()
            for k, v in (("arc", np.asarray(a_)), ("gauss", np.asarray(g_)))},
         s_vabs=sVg, s_v2=sM2, s_first=sM1)
open(os.path.join(DATA, "README_v2_paths.txt"), "w").write("\n".join(rep) + "\n")
print("figures: r020_paths_v2_{circ,cap,junc}.png", flush=True)
