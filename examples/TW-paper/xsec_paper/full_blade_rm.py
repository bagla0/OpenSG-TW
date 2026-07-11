"""full_blade_rm.py -- FULL IEA-22 blade: RM 6-DOF ring vs 2-D solid at every span
station (r=0.2..0.9).  ONLY the RM 6-DOF constrained ring (no KL, no strip-RM).
Prints per-station full 6x6 %err, a spanwise diagonal summary, and saves a spanwise
%err plot + npz.

    python full_blade_rm.py
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for q in (MITC, REPO):
    sys.path.insert(0, q)
from xsec_5v6_master import load_ring, load_solid, ring_6dof, LBL

IB = os.path.abspath(os.path.join(HERE, "..", "iea22_blade", "data"))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
STATIONS = [("r020", 0.2), ("r030", 0.3), ("r040", 0.4), ("r050", 0.5),
            ("r060", 0.6), ("r070", 0.7), ("r080", 0.8), ("r090", 0.9)]


def pct(S, Rf):
    cut = np.abs(Rf).max() / 1e3
    E = np.full((6, 6), np.nan)
    m = np.abs(Rf) > cut
    E[m] = 100.0 * (S[m] - Rf[m]) / Rf[m]
    return E


rows, diag_err, tt = [], [], []
for tag, r in STATIONS:
    shell = os.path.join(IB, "shell_%s.yaml" % tag)
    solid = os.path.join(IB, "C6_solid_%s.txt" % tag)
    if not (os.path.exists(shell) and os.path.exists(solid)):
        print("skip", tag); continue
    So = load_solid(solid)
    t0 = time.time(); C6 = ring_6dof(load_ring(shell)); dt = time.time() - t0
    tt.append(dt)
    e = [100.0 * (C6[i, i] - So[i, i]) / So[i, i] for i in range(6)]
    diag_err.append(e)
    rows.append((tag, r, So, C6))
    print("r=%.1f  RM6DOF diag %%err: %s   [%.2fs]"
          % (r, "  ".join("%s%+6.2f" % (LBL[i], e[i]) for i in range(6)), dt), flush=True)

R = [r for _t, r, _s, _c in rows]
E = np.array(diag_err)
np.savez(os.path.join(OUT, "full_blade_rm.npz"),
         r=np.array(R), diag_err=E, labels=LBL,
         solids=np.array([s for _t, _r, s, _c in rows]),
         rings=np.array([c for _t, _r, _s, c in rows]), times=np.array(tt))

# spanwise %err plot (RM 6-DOF only): shear+torsion panel and classical panel
MARK = ["o", "s", "^", "D", "v", "P"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
groups = [([1, 2, 3], "transverse shear & torsion"), ([0, 4, 5], "extension & bending")]
for a, (idx, title) in zip(ax, groups):
    a.axhline(0, color="0.6", lw=0.8)
    for k in idx:
        a.plot(R, E[:, k], color=COL[k], marker=MARK[k], lw=1.6, ms=5, label="$%s$" %
               ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"][k])
    a.set_xlabel("span station  $r$"); a.set_title(title); a.grid(alpha=0.3)
    a.legend(fontsize=9, frameon=False)
ax[0].set_ylabel("diagonal % error vs 2-D solid")
fig.tight_layout(); p = os.path.join(FIG, "full_blade_rm_span.png")
fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
print("\nwrote", p)
print("mean ring time (stations 2-8): %.2f s" % float(np.mean(tt[1:])))
