"""msgrm_r020_plots.py -- line + contour plots of the MSG-RM dehom vs VABS at r=0.2 (iea_s10).

Reads the variant fields from _msgrm_variants.npz (sV2 = +web registration, sV3 = +registration
+MSG-RM first-order recovery) and the VABS .SM, and renders:
  figures/r020_msgrm_web_tt.png    sigma12 through the web thickness (per web, mid-height band):
                                   VABS vs RM baseline vs MSG-RM (registration visible)
  figures/r020_msgrm_circ.png      circumferential sigma11 / sigma12 (element means along the skin
                                   loop, normalized arc): VABS vs baseline vs MSG-RM
  figures/r020_msgrm_cap_tt.png    spar-cap centre through-thickness sigma11 + sigma12: VABS vs MSG-RM
  figures/r020_msgrm_contours.png  sigma11/22/12 contours VABS | MSG-RM (exploded, clamped)
Conventions: VABS blue #1f77b4 solid/filled, RM orange #ff7f0e dashed/open squares, baseline gray;
non-dimensional abscissa (zeta 0=OML->1=IML, arc s/S); fonts 15/17/14; no titles.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "legend.fontsize": 14,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})
BLUE = "#1f77b4"; ORANGE = "#ff7f0e"; GRAY = "0.55"

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
sys.path.insert(0, XSEC)
VABS = os.path.join(IEA, "out", "VABS_iea51")
FIG = os.path.join(HERE, "figures")

import yaml as _yaml

d = _yaml.safe_load(open(os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml")))

dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]
sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6
sV1 = np.load(os.path.join(HERE, "_rm_s10_cache.npz"))["sRg"]
z = np.load(os.path.join(HERE, "_msgrm_variants.npz"))
sV2, sV3, zoff, is_web = z["sV2"], z["sV3"], z["zoff"], z["is_web"]

# ---- ring geometry (corners/cells) straight from the yaml, center ref ----
from oml_ring import load_ring_ref

R = load_ring_ref(os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml"), "center")
rc = np.asarray(R["cells"]); corners = np.asarray(R["rx"])[:, R["cross"]]
cen = corners.mean(0)
sec_names = [s["elementSet"] for s in d["sections"]]
lay_of = [sec_names[int(si)] for si in R["rsub"]]
hth = {s["elementSet"]: float(sum(float(p[1]) for p in s["layup"])) for s in d["sections"]}
n_el = len(rc)
emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])

# assign every gauss point to the nearest ring element midline + local depth
kd = cKDTree(emid)
el = kd.query(sm_xy)[1]
tvec = corners[rc[:, 1]] - corners[rc[:, 0]]
tlen = np.linalg.norm(tvec, axis=1); tun = tvec / tlen[:, None]
nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
flip = (cen - emid)[:, 0] * nvec[:, 0] + (cen - emid)[:, 1] * nvec[:, 1] < 0
nvec[flip] *= -1.0
dp = sm_xy - corners[rc[el, 0]]
zdep = dp[:, 0] * nvec[el, 0] + dp[:, 1] * nvec[el, 1]

# ---- web chains (ordered) ----
deg = np.zeros(int(rc.max()) + 1, int)
for a, b in rc:
    deg[a] += 1; deg[b] += 1
adjm = {}
for e, (a, b) in enumerate(rc):
    adjm.setdefault(a, []).append((b, e)); adjm.setdefault(b, []).append((a, e))
junc = set(np.where(deg >= 3)[0])
chains, seen = [], set()
for j in junc:
    for (nxt, e0) in adjm[j]:
        if e0 in seen:
            continue
        chain, prev, cur = [e0], j, nxt
        seen.add(e0)
        while cur not in junc and deg[cur] == 2:
            (n1, e1), (n2_, e2) = adjm[cur][0], adjm[cur][1]
            nn2, ee = (n1, e1) if n1 != prev else (n2_, e2)
            if ee in seen:
                break
            chain.append(ee); seen.add(ee)
            prev, cur = cur, nn2
        if cur in junc and is_web[chain].all():
            chains.append(chain)
chains = sorted(chains, key=lambda c: emid[c].mean(0)[0])

# ================= 1: sigma12 through the WEB thickness =================
fig, ax = plt.subplots(1, len(chains), figsize=(6.2 * len(chains), 5.2), sharey=False)
ax = np.atleast_1d(ax)
for wi, chain in enumerate(chains):
    ce = np.array(chain)
    m = np.isin(el, ce)
    y3 = sm_xy[m, 1]
    ymid = 0.5 * (y3.min() + y3.max()); span = y3.max() - y3.min()
    mm = m.copy(); mm[m] = np.abs(y3 - ymid) < 0.12 * span
    e_of = el[mm]
    h = np.array([hth[lay_of[e]] for e in e_of])
    zc = zdep[mm] - zoff[e_of]
    zeta = np.clip((zc + h / 2) / h, -0.15, 1.15)
    o = np.argsort(zeta)
    ax[wi].plot(zeta[o], sVg[mm, 5][o], "o", color=BLUE, ms=6.5, label="VABS")
    ax[wi].plot(zeta[o], sV1[mm, 5][o], "x", color=GRAY, ms=6.0, label="RM baseline")
    ax[wi].plot(zeta[o], sV3[mm, 5][o], "s", mfc="none", color=ORANGE, ms=7.0, label="MSG-RM")
    ax[wi].set_xlabel(r"$\zeta$ through web %d thickness" % wi)
    ax[wi].set_ylabel(r"$\sigma_{12}$ (MPa)" if wi == 0 else "")
    ax[wi].grid(alpha=0.3)
ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r020_msgrm_web_tt.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote r020_msgrm_web_tt.png")

# ================= 2: circumferential element-mean sigma11 / sigma12 =================
skin = np.where(~is_web)[0]
ang = np.arctan2(emid[skin, 1] - cen[1], emid[skin, 0] - cen[0])
order = skin[np.argsort(ang)]
L = tlen[order]
s_arc = (np.cumsum(L) - 0.5 * L); s_arc /= s_arc[-1] + 0.5 * L[-1] / s_arc[-1] if s_arc[-1] else 1.0
s_arc = s_arc / s_arc.max()


def elem_mean(field):
    out = np.full(len(order), np.nan)
    for i, e in enumerate(order):
        m = el == e
        if m.any():
            out[i] = field[m].mean()
    return out


fig, ax = plt.subplots(2, 1, figsize=(13.5, 8.4), sharex=True)
for r_, (ci, lab) in enumerate([(0, r"$\sigma_{11}$ (MPa)"), (5, r"$\sigma_{12}$ (MPa)")]):
    ax[r_].plot(s_arc, elem_mean(sVg[:, ci]), "-", color=BLUE, lw=2.2, label="VABS")
    ax[r_].plot(s_arc, elem_mean(sV1[:, ci]), "-", color=GRAY, lw=1.1, label="RM baseline")
    ax[r_].plot(s_arc, elem_mean(sV3[:, ci]), "--s", color=ORANGE, ms=4.5, mfc="none", lw=1.4,
                markevery=4, label="MSG-RM")
    ax[r_].set_ylabel(lab); ax[r_].grid(alpha=0.3)
ax[1].set_xlabel(r"normalized skin arc $s/S$")
ax[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r020_msgrm_circ.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote r020_msgrm_circ.png")

# ================= 3: spar-cap centre through-thickness =================
capels = [e for e in skin if abs(emid[e, 0]) < 0.35 and emid[e, 1] > 0 and lay_of[e].endswith("2")]
if not capels:
    capels = [e for e in skin if abs(emid[e, 0]) < 0.35 and emid[e, 1] > 0]
ecap = capels[len(capels) // 2]
m = el == ecap
h = hth[lay_of[ecap]]
zeta = np.clip((zdep[m] + h / 2) / h, -0.1, 1.1)
o = np.argsort(zeta)
fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2))
for k, (ci, lab) in enumerate([(0, r"$\sigma_{11}$ (MPa)"), (5, r"$\sigma_{12}$ (MPa)")]):
    ax[k].plot(zeta[o], sVg[m, ci][o], "o-", color=BLUE, ms=6.5, lw=1.4, label="VABS")
    ax[k].plot(zeta[o], sV3[m, ci][o], "--s", color=ORANGE, ms=7.0, mfc="none", lw=1.4, label="MSG-RM")
    ax[k].set_xlabel(r"$\zeta$ (0=OML, 1=IML), cap centre")
    ax[k].set_ylabel(lab); ax[k].grid(alpha=0.3)
ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r020_msgrm_cap_tt.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote r020_msgrm_cap_tt.png (elem %d layup %s)" % (ecap, lay_of[ecap]))

# ================= 4: contours sigma11/22/12 VABS | MSG-RM =================
U = np.loadtxt(os.path.join(VABS, "iea_s10.sg.U")); U = U[np.argsort(U[:, 0])]
xy = U[:, 1:3]
L2 = [l for l in open(os.path.join(VABS, "iea_s10.sg")).read().splitlines() if l.strip()]
hi = next(i for i, l in enumerate(L2) if len(l.split()) == 3
          and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
nn, ne, nm = [int(x) for x in L2[hi].split()]
conn = [[int(x) for x in L2[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
trl = []
for c in conn:
    c0 = [n - 1 for n in c]
    if len(c0) == 3:
        trl.append(c0[:3])
    elif len(c0) >= 4:
        trl.extend([[c0[0], c0[1], c0[2]], [c0[0], c0[2], c0[3]]])
tris = np.array(trl); M = tris.shape[0]
gxy = sm_xy.reshape(M, 3, 2)
Ai = np.linalg.inv(np.concatenate([np.ones((M, 3, 1)), gxy], 2))
Cc = np.concatenate([np.ones((M, 3, 1)), xy[tris]], 2)
EP = xy[tris].reshape(-1, 2)
etri = mtri.Triangulation(EP[:, 0], EP[:, 1], np.arange(3 * M).reshape(M, 3))


def corners_of(val):
    g = val.reshape(M, 3, 1)
    c = Cc @ (Ai @ g)
    return np.clip(c, g.min(1, keepdims=True), g.max(1, keepdims=True)).reshape(-1)


fig, ax = plt.subplots(3, 2, figsize=(13.5, 12.0))
for r_, (ci, lab) in enumerate([(0, r"$\sigma_{11}$"), (1, r"$\sigma_{22}$"), (5, r"$\sigma_{12}$")]):
    mlim = np.nanpercentile(np.abs(np.r_[sVg[:, ci], sV3[:, ci]]), 99) or 1e-9
    for c_, val in enumerate([sVg[:, ci], sV3[:, ci]]):
        cs = ax[r_, c_].tripcolor(etri, np.clip(corners_of(val), -mlim, mlim), shading="gouraud",
                                  cmap="rainbow", vmin=-mlim, vmax=mlim)
        ax[r_, c_].set_aspect("equal"); ax[r_, c_].axis("off")
    ax[r_, 0].text(-0.05, 0.5, lab, transform=ax[r_, 0].transAxes, rotation=90,
                   va="center", ha="right", fontsize=20)
    cb = fig.colorbar(cs, ax=ax[r_, :].tolist(), shrink=0.85, pad=0.015)
    cb.set_label("MPa", fontsize=18); cb.ax.tick_params(labelsize=15)
ax[0, 0].set_title("VABS", fontsize=20, pad=8)
ax[0, 1].set_title("MSG-RM shell", fontsize=20, pad=8)
fig.savefig(os.path.join(FIG, "r020_msgrm_contours.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote r020_msgrm_contours.png")
