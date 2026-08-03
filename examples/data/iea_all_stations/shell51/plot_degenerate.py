'''plot_degenerate.py -- show WHY PreVABS fails on the two IEA sections, on the 1-D OML contour,
and what the densify / TE-open fix does. Legend OUTSIDE. No title.
Row 1 (s02): OML + baseline division points; the ply inward-offset (~wall thickness) is many x the
shortest baseline segment -> that segment collapses (offset.cpp "base must have at least 2 vertices").
Row 2 (s50): the sub-metre tip OML + its collapsing sharp trailing edge (PDCEL sliver-TE walk).
Right column = the fix (densified baseline / opened TE).'''
import os
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
XMLD = os.path.join(HERE, 'xml')


def read_dat(name):
    pts = []
    for ln in open(os.path.join(XMLD, name)):
        s = ln.split()
        if len(s) == 2:
            try:
                pts.append([float(s[0]), float(s[1])])
            except ValueError:
                pass
    return np.array(pts)


def load(tag):
    root = ET.parse(os.path.join(XMLD, 'iea_%s.xml' % tag)).getroot()
    scale = float(root.find('general/scale').text)
    datn = root.find(".//line[@type='airfoil']/points").text.strip()
    xy = read_dat(datn) * scale                    # OML in metres, TE->LE->TE
    # division points by normalized x2 on top/bottom
    dpts = []
    for p in root.findall("baselines/point[@on]"):
        xn = float(p.text) * scale
        dpts.append((xn, p.get('which', 'top')))
    # web feet (wp)
    webs = []
    for p in root.findall('baselines/point'):
        if (p.get('name') or '').startswith('wp'):
            t = [float(v) * scale for v in p.text.split()]
            webs.append(t)
    return xy, scale, dpts, webs


def split_tb(xy):
    le = int(np.argmin(xy[:, 0]))
    return xy[:le + 1], xy[le:]                     # top (TE->LE), bottom (LE->TE)


def dp_xy(xy, xn, which):
    top, bot = split_tb(xy)
    seg = top if which == 'top' else bot
    j = int(np.argmin(np.abs(seg[:, 0] - xn)))
    return seg[j]


fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.2))

# ---------- Row 1: s02 offset degeneracy ----------
xy, sc, dpts, webs = load('s02')
P = np.vstack([xy, xy[0]])
for c in (0, 1):
    ax[0, c].plot(P[:, 0], P[:, 1], '-', color='0.6', lw=1.3, zorder=1)
    for w in webs:
        ax[0, c].plot(w[0], w[1], 'v', color='tab:red', ms=5, zorder=2)
    ax[0, c].set_aspect('equal'); ax[0, c].set_xlabel('x2 (chordwise) [m]'); ax[0, c].set_ylabel('x3 [m]')
# division points + segment lengths
D = np.array([dp_xy(xy, xn, w) for (xn, w) in dpts])
order = np.argsort(np.arctan2(D[:, 1] - D[:, 1].mean(), D[:, 0] - D[:, 0].mean()))
seglen = []
for k in range(len(dpts)):
    a, b = D[k], D[(k + 1) % len(dpts)]
    seglen.append(np.hypot(*(b - a)))
seglen = np.array(seglen)
kmin = int(np.argmin(seglen))
tmid = 0.06                                         # representative ply/laminate inward offset [m]
ax[0, 0].plot(D[:, 0], D[:, 1], 'o', color='navy', ms=4, zorder=3)
a, b = D[kmin], D[(kmin + 1) % len(dpts)]
ax[0, 0].plot([a[0], b[0]], [a[1], b[1]], '-', color='crimson', lw=3, zorder=4)
mid = 0.5 * (a + b)
nrm = np.array([-(b - a)[1], (b - a)[0]]); nrm = nrm / (np.hypot(*nrm) + 1e-9)
if np.dot(nrm, D.mean(0) - mid) < 0:
    nrm = -nrm
ax[0, 0].annotate('', xy=mid + tmid * nrm, xytext=mid,
                  arrowprops=dict(arrowstyle='->', color='crimson', lw=1.6), zorder=5)
seg_txt = ('%.4f m' % seglen[kmin]) if seglen[kmin] > 1e-4 else '≈ 0 (two division pts coincide)'
ratio_txt = ('%.0f×' % (tmid / seglen[kmin])) if seglen[kmin] > 1e-4 else '>40× (PreVABS log: 0.0015 m seg)'
ax[0, 0].text(0.02, 0.97, 's02 (η=0.04): PreVABS FAILS\n'
              'shortest baseline seg = %s\n'
              'ply inward offset ≈ %.3f m  (%s the seg)\n'
              '→ segment collapses to 1 vertex\n'
              '(offset.cpp: "base must have ≥ 2 vertices")'
              % (seg_txt, tmid, ratio_txt),
              transform=ax[0, 0].transAxes, va='top', fontsize=9, color='0.15',
              bbox=dict(boxstyle='round', fc='#fff3f3', ec='crimson'))
# solution: densify the short segment
dens = np.linspace(a, b, 6)
ax[0, 1].plot(D[:, 0], D[:, 1], 'o', color='navy', ms=4, zorder=3)
ax[0, 1].plot(dens[:, 0], dens[:, 1], 's', color='tab:green', ms=6, zorder=5)
ax[0, 1].annotate('', xy=mid + tmid * nrm, xytext=mid,
                  arrowprops=dict(arrowstyle='->', color='tab:green', lw=1.6), zorder=5)
ax[0, 1].text(0.02, 0.97, 'FIX: densify baseline so every\n'
              'segment > ply offset (add points, green)\n'
              '→ offset stays valid, PreVABS meshes it',
              transform=ax[0, 1].transAxes, va='top', fontsize=9, color='0.15',
              bbox=dict(boxstyle='round', fc='#f2fbf2', ec='tab:green'))

# ---------- Row 2: s50 sliver TE ----------
xy2, sc2, dpts2, webs2 = load('s50')
P2 = np.vstack([xy2, xy2[0]])
te = xy2[np.argmax(xy2[:, 0])]                      # TE point (max x2)
for c in (0, 1):
    ax[1, c].plot(P2[:, 0], P2[:, 1], '-', color='0.6', lw=1.3, zorder=1)
    ax[1, c].set_aspect('equal'); ax[1, c].set_xlabel('x2 (chordwise) [m]'); ax[1, c].set_ylabel('x3 [m]')
    # zoom on the TE
    ax[1, c].set_xlim(te[0] - 0.10 * sc2 * 0 - 0.06, te[0] + 0.01)
    ax[1, c].set_ylim(-0.05, 0.05)
# problem: near-zero TE angle
near = xy2[np.abs(xy2[:, 0] - te[0]) < 0.06]
ax[1, 0].plot(near[:, 0], near[:, 1], '.-', color='crimson', lw=1.5, ms=4, zorder=4)
ax[1, 0].plot(te[0], te[1], 'x', color='crimson', ms=10, zorder=5)
ax[1, 0].text(0.02, 0.97, 's50 (η=1.0 tip, chord=%.2f m): PreVABS FAILS\n'
              'the two skins meet at the TE at ≈1e-13 rad\n'
              '→ collapsing sliver; the TE loop cannot close\n'
              '(PDCEL: walkLoopWithLimit / near-degenerate angles)' % sc2,
              transform=ax[1, 0].transAxes, va='top', fontsize=9, color='0.15',
              bbox=dict(boxstyle='round', fc='#fff3f3', ec='crimson'))
# solution: open the TE (finite gap)
gap = 0.010 * sc2
tt = np.array([te[0], te[1] + gap]); tb = np.array([te[0], te[1] - gap])
ax[1, 1].plot(near[:, 0], near[:, 1], '.-', color='0.6', lw=1.2, ms=3, zorder=3)
ax[1, 1].plot([tt[0], tb[0]], [tt[1], tb[1]], '-', color='tab:green', lw=3, zorder=5)
ax[1, 1].text(0.02, 0.97, 'FIX: open/blunt the TE with a small\n'
              'finite gap (manageTE, green) → valid loop,\n'
              'PreVABS meshes the tip (or use shell-only)',
              transform=ax[1, 1].transAxes, va='top', fontsize=9, color='0.15',
              bbox=dict(boxstyle='round', fc='#f2fbf2', ec='tab:green'))

# legend OUTSIDE (bottom), never over the plots
handles = [Line2D([0], [0], color='0.6', lw=1.3, label='OML contour'),
           Line2D([0], [0], marker='o', color='navy', lw=0, label='baseline division points'),
           Line2D([0], [0], color='crimson', lw=3, label='degenerate feature (fails)'),
           Line2D([0], [0], marker='s', color='tab:green', lw=0, label='fix: densified pts / opened TE'),
           Line2D([0], [0], marker='v', color='tab:red', lw=0, label='shear-web foot')]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.04, 1, 1])
out = os.path.join(HERE, 'degenerate_diagnosis.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print('wrote', out)
print('s02 shortest seg=%.5f m (coincident if ~0)' % seglen[kmin])
print('s50 chord=%.3f m' % sc2)
