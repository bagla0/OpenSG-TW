"""msgrm_r020_pcterr.py -- percent-error plots of the MSG-RM dehom vs VABS at r=0.2.

Two paths, from data/msgrm_r020_fields.npz:
  (a) LP (suction-side) spar-cap LEFT-edge through-thickness column (the leftmost
      suction cap element): sigma11/sigma12 and u1/u2/u3 error vs zeta.
  (b) circumferential skin loop (element means, normalized arc s/S).
Error definition (robust where the signal crosses zero):
      err% = 100 * (MSG-RM - VABS) / max_path |VABS|   (per component).
Outputs figures/r020_pcterr_capleft.png and figures/r020_pcterr_circ.png.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "legend.fontsize": 14,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})
ORANGE = "#ff7f0e"; BLUE = "#1f77b4"; GREEN = "#2ca02c"

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
sys.path.insert(0, XSEC)
import yaml as _yaml
from oml_ring import load_ring_ref

z = np.load(os.path.join(HERE, "data", "msgrm_r020_fields.npz"))
sm_xy = z["sm_xy"]; sM = z["stress_msgrm"]; sV = z["stress_vabs"]
xyn = z["xy_nodes"]; uM = z["disp_msgrm"]; uV = z["disp_vabs"]
elg = z["el_gauss"]; zdg = z["z_gauss"]; eln = z["el_node"]; zdn = z["z_node"]
is_web = z["is_web"]

SHELL = os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml")
d = _yaml.safe_load(open(SHELL))
R = load_ring_ref(SHELL, "center")
rc = np.asarray(R["cells"]); corners = np.asarray(R["rx"])[:, R["cross"]]
sec_names = [s["elementSet"] for s in d["sections"]]
lay_of = [sec_names[int(si)] for si in R["rsub"]]
hth = {s["elementSet"]: float(sum(float(p[1]) for p in s["layup"])) for s in d["sections"]}
emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
Lel = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)
cen = corners.mean(0)

# ---------- (a) LP spar-cap LEFT edge through-thickness ----------
cap = [e for e in range(len(rc)) if lay_of[e] == "layup_2" and emid[e, 1] > 0]
e_left = cap[int(np.argmin([emid[e, 0] for e in cap]))]
h = hth["layup_2"]
mg = elg == e_left
mn = eln == e_left
zeta_g = np.clip((zdg[mg] + h / 2) / h, -0.05, 1.05)
zeta_n = np.clip((zdn[mn] + h / 2) / h, -0.05, 1.05)
og = np.argsort(zeta_g); on = np.argsort(zeta_n)
print("cap-left element %d at x2=%.3f y3=%.3f, %d gauss / %d nodes"
      % (e_left, emid[e_left, 0], emid[e_left, 1], mg.sum(), mn.sum()))

fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.2))
for ci, lab, col in ((0, r"$\sigma_{11}$", BLUE), (5, r"$\sigma_{12}$", ORANGE)):
    ref = np.abs(sV[mg, ci]).max()
    pct = 100.0 * (sM[mg, ci] - sV[mg, ci]) / max(ref, 1e-12)
    ax[0].plot(zeta_g[og], pct[og], "o-", color=col, ms=5.0, lw=1.2, label=lab)
ax[0].set_xlabel(r"$\zeta$ (0=OML, 1=IML), LP cap left edge")
ax[0].set_ylabel(r"stress error (\% of path peak $|$VABS$|$)")
ax[0].grid(alpha=0.3); ax[0].legend(frameon=False)
for ci, lab, col in ((0, r"$u_1$", BLUE), (1, r"$u_2$", ORANGE), (2, r"$u_3$", GREEN)):
    ref = np.abs(uV[mn, ci]).max()
    pct = 100.0 * (uM[mn, ci] - uV[mn, ci]) / max(ref, 1e-12)
    ax[1].plot(zeta_n[on], pct[on], "s-", color=col, ms=5.0, lw=1.2, mfc="none", label=lab)
ax[1].set_xlabel(r"$\zeta$ (0=OML, 1=IML), LP cap left edge")
ax[1].set_ylabel(r"disp error (\% of path peak $|$VABS$|$)")
ax[1].grid(alpha=0.3); ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "r020_pcterr_capleft.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote figures/r020_pcterr_capleft.png")

# ---------- (b) circumferential (skin loop, element means) ----------
skin = np.where(~is_web)[0]
ang = np.arctan2(emid[skin, 1] - cen[1], emid[skin, 0] - cen[0])
order = skin[np.argsort(ang)]
s_arc = np.cumsum(Lel[order]) - 0.5 * Lel[order]; s_arc /= s_arc.max()


def elem_mean(field, els):
    out = np.full(len(order), np.nan)
    for i, e in enumerate(order):
        m = els == e
        if m.any():
            out[i] = field[m].mean()
    return out


fig, ax = plt.subplots(2, 1, figsize=(13.5, 8.6), sharex=True)
for ci, lab, col in ((0, r"$\sigma_{11}$", BLUE), (5, r"$\sigma_{12}$", ORANGE)):
    v = elem_mean(sV[:, ci], elg); m_ = elem_mean(sM[:, ci], elg)
    ref = np.nanmax(np.abs(v))
    ax[0].plot(s_arc, 100.0 * (m_ - v) / ref, "-", color=col, lw=1.6, label=lab)
ax[0].set_ylabel(r"stress error (\% of loop peak)"); ax[0].grid(alpha=0.3)
ax[0].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
for ci, lab, col in ((0, r"$u_1$", BLUE), (1, r"$u_2$", ORANGE), (2, r"$u_3$", GREEN)):
    v = elem_mean(uV[:, ci], eln); m_ = elem_mean(uM[:, ci], eln)
    ref = np.nanmax(np.abs(v))
    ax[1].plot(s_arc, 100.0 * (m_ - v) / ref, "-", color=col, lw=1.6, label=lab)
ax[1].set_ylabel(r"disp error (\% of loop peak)"); ax[1].set_xlabel(r"normalized skin arc $s/S$")
ax[1].grid(alpha=0.3); ax[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "r020_pcterr_circ.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote figures/r020_pcterr_circ.png")
