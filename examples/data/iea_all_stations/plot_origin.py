"""plot_origin.py -- show WHERE the 1-D shell cross-section origin sits.
Left: r0247 contour (OML vs webs) with the CURRENT origin (0,0) and the windIO reference-axis
point (pitch_axis*chord from LE) marked.  Right: chordwise LE->reference-axis offset vs span,
proving the LE is NOT a consistent beam reference (it sweeps), while the reference axis is."""
import os
from collections import defaultdict

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
YDIR = os.path.join(HERE, "1d_yaml")
WINDIO = os.path.join(HERE, "IEA-22-280-RWT.yaml")
ETA = {'r0000': 0.0, 'r0020': 0.02, 'r0049': 0.04868313468735217, 'r0066': 0.06649308419703373,
       'r0083': 0.08345591801468194, 'r0102': 0.10220904193570582, 'r0110': 0.11036272014164386,
       'r0136': 0.1364246061019374, 'r0156': 0.15564440587515185, 'r0197': 0.19665336575444797,
       'r0247': 0.24696148735364706, 'r0399': 0.3992636115637571, 'r0534': 0.5335887750152993,
       'r0739': 0.738938689884722, 'r0980': 0.9799991709122947, 'r1000': 1.0}


def rows(raw, cast):
    out = []
    for r in raw:
        toks = r.split() if isinstance(r, str) else (r[0].split() if (len(r) == 1 and isinstance(r[0], str)) else r)
        out.append([cast(t) for t in toks])
    return np.array(out)


def outer_loop(nd, el):
    adj = defaultdict(set)
    for a, b in el:
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    xy = nd[:, :2]
    start = int(np.argmin(xy[:, 0])) + 1
    prev_node, cur = None, start
    din = np.array([0.0, -1.0]); loop = []
    for _ in range(2 * len(el) + 5):
        v = xy[cur - 1]
        cand = [w for w in adj[cur] if w != prev_node and w != cur]
        if not cand:
            break
        angs = [np.arctan2(din[0] * (xy[w - 1] - v)[1] - din[1] * (xy[w - 1] - v)[0],
                           din[0] * (xy[w - 1] - v)[0] + din[1] * (xy[w - 1] - v)[1]) for w in cand]
        nxt = cand[int(np.argmin(angs))]
        loop.append(cur); prev_node = cur; din = xy[nxt - 1] - v; cur = nxt
        if cur == start:
            break
    return np.array(loop)


# windIO section_offset_y (LE->reference-axis chordwise distance) + chord
d = yaml.safe_load(open(WINDIO))
osh = d['components']['blade']['outer_shape']
so = osh['section_offset_y']; ch = osh['chord']
so_g, so_v = np.array(so['grid']), np.array(so['values'])
ch_g, ch_v = np.array(ch['grid']), np.array(ch['values'])


def refaxis_x2(e): return float(np.interp(e, so_g, so_v))   # LE->ref-axis chordwise distance [m]
def chord(e): return float(np.interp(e, ch_g, ch_v))
def pitch_axis(e): return refaxis_x2(e) / chord(e)          # as a chord fraction (for labels)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- LEFT: r0247 contour with origin markers
tag = 'r0247'; e = ETA[tag]
dd = yaml.safe_load(open(os.path.join(YDIR, 'iea_%s_shell.yaml' % tag)))
nd = rows(dd['nodes'], float); el = rows(dd['elements'], int)
# all elements (draw webs faint)
for a, b in el:
    ax.plot([nd[a - 1, 0], nd[b - 1, 0]], [nd[a - 1, 1], nd[b - 1, 1]], '-', color='0.75', lw=0.8, zorder=1)
lp = outer_loop(nd, el); c = nd[lp - 1]; cc = np.vstack([c, c[0]])
ax.plot(cc[:, 0], cc[:, 1], '-', color='tab:green', lw=2.2, zorder=3, label='OML (skins)')
# webs = elements not on the OML loop
onloop = set(map(int, lp))
first = True
for a, b in el:
    if not (int(a) in onloop and int(b) in onloop):
        ax.plot([nd[a - 1, 0], nd[b - 1, 0]], [nd[a - 1, 1], nd[b - 1, 1]], '-', color='tab:red',
                lw=1.6, zorder=2, label=('shear webs' if first else None)); first = False

pax = refaxis_x2(e)                                 # chordwise LE->ref-axis distance (section_offset_y)
ax.axhline(0, color='0.6', ls=':', lw=1, zorder=0)
ax.plot(0, 0, 'o', color='crimson', ms=13, zorder=5, label='CURRENT origin (0,0) = LE')
ax.plot(pax, 0, '*', color='navy', ms=20, zorder=5, label='windIO reference axis (pitch axis)')
ax.annotate('LE\n(current origin)', (0, 0), textcoords='offset points', xytext=(6, 14),
            color='crimson', fontsize=9, weight='bold')
ax.annotate('reference axis\n%.0f%% chord = %.2f m' % (100 * pitch_axis(e), pax), (pax, 0),
            textcoords='offset points', xytext=(6, -34), color='navy', fontsize=9, weight='bold')
ax.annotate('', xy=(pax, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='<->', color='0.3', lw=1.4))
ax.text(pax / 2, 0.18, 'offset = %.2f m' % pax, ha='center', color='0.2', fontsize=9)
ax.set_xlabel('x2  (chordwise) [m]'); ax.set_ylabel('x3  (flapwise) [m]')
ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
ax.text(0.02, 0.03, 'IEA r0247 (eta=0.247), chord=%.2f m' % chord(e), transform=ax.transAxes,
        fontsize=9, color='0.3')

# ---- RIGHT: LE->ref-axis chordwise offset vs span  (the piecewise-beam-line problem)
tags = sorted(ETA, key=lambda t: ETA[t])
es = np.array([ETA[t] for t in tags])
offs = np.array([refaxis_x2(e) for e in es])
L = 138.204
ax2.plot(es, np.zeros_like(es), '-', color='crimson', lw=2, marker='o', ms=4,
         label='LE line (current origin) = straight only by construction')
ax2.plot(es, offs, '-', color='navy', lw=2, marker='*', ms=9,
         label='windIO reference axis, in the LE frame')
ax2.fill_between(es, 0, offs, color='navy', alpha=0.08)
ax2.set_xlabel('span fraction  eta'); ax2.set_ylabel('chordwise position in LE-origin frame [m]')
ax2.legend(loc='upper right', fontsize=8.5)
ax2.text(0.02, 0.95, 'If LE=origin, the TRUE reference axis sweeps 0 -> %.1f m -> 0\n'
         '=> a zigzag beam line. Putting the origin ON the reference axis\n'
         'makes x1 straight (that is the fix).' % offs.max(),
         transform=ax2.transAxes, va='top', fontsize=9, color='0.2',
         bbox=dict(boxstyle='round', fc='0.96', ec='0.7'))
ax2.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(HERE, 'origin_1d_cross_section.png')
fig.savefig(out, dpi=140)
print('wrote', out)
print('r0247: pitch_axis=%.4f  chord=%.3f  LE->refaxis=%.3f m' % (pitch_axis(e), chord(e), pax))
print('max LE->refaxis offset over span = %.3f m' % offs.max())
