'''plot_orientation.py -- render each 1-D shell cross-section (mid-surface, reference-axis origin)
with its material-frame arrows: e2 = blue (in-plane ply direction), e3 = black (wall normal, OML->IML).
Mesh coloured by layup (rainbow); origin (0,0) = windIO reference axis marked; ALL legends OUTSIDE the
plot (never covering the data).  One PNG per station -> orientation_png/iea_r*_orient.png.
    ~/miniconda3/envs/opensg_2_0/bin/python plot_orientation.py            # all stations
    ~/miniconda3/envs/opensg_2_0/bin/python plot_orientation.py r0247      # one station
'''
import glob
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
YDIR = os.path.join(HERE, '1d_yaml')
OUT = os.path.join(HERE, 'orientation_png'); os.makedirs(OUT, exist_ok=True)


def rows(raw, cast):
    out = []
    for r in raw:
        toks = r.split() if isinstance(r, str) else (r[0].split() if (len(r) == 1 and isinstance(r[0], str)) else r)
        out.append([cast(t) for t in toks])
    return np.array(out)


def plot_one(path):
    tag = os.path.basename(path).split('_')[1]
    d = yaml.safe_load(open(path))
    nd = rows(d['nodes'], float)
    el = rows(d['elements'], int) - 1
    ori = rows(d['elementOrientations'], float)          # [e1(3) e2(3) e3(3)] per element
    e2, e3 = ori[:, 3:6], ori[:, 6:9]
    # element -> layup set
    es = d['sets']['element']
    setid = np.zeros(len(el), int)
    lab2set = {int(lab): si for si, s in enumerate(es) for lab in s['labels']}
    for k in range(len(el)):
        setid[k] = lab2set.get(k + 1, 0)
    nsets = len(es)
    cols = plt.cm.rainbow(np.linspace(0, 1, max(nsets, 2)))

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    # mesh coloured by layup
    seen = set()
    for k, (a, b) in enumerate(el):
        c = cols[setid[k]]
        lab = ('layup_%d' % setid[k]) if setid[k] not in seen else None
        seen.add(setid[k])
        ax.plot([nd[a, 0], nd[b, 0]], [nd[a, 1], nd[b, 1]], '-', color=c, lw=1.6, label=lab, zorder=2)

    # e2/e3 arrows at element midpoints, subsampled for clarity
    mid = 0.5 * (nd[el[:, 0]] + nd[el[:, 1]])
    span = max(np.ptp(nd[:, 0]), np.ptp(nd[:, 1]))
    L = 0.045 * span                                     # arrow length
    step = max(1, len(el) // 44)
    for k in range(0, len(el), step):
        m = mid[k]
        ax.annotate('', xy=(m[0] + L * e2[k, 0], m[1] + L * e2[k, 1]), xytext=(m[0], m[1]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.0), zorder=4)
        ax.annotate('', xy=(m[0] + L * e3[k, 0], m[1] + L * e3[k, 1]), xytext=(m[0], m[1]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.0), zorder=4)

    ax.plot(0, 0, 'P', color='crimson', ms=12, zorder=6)
    ax.annotate('reference axis x1 (origin)', (0, 0), textcoords='offset points', xytext=(8, 8),
                color='crimson', fontsize=8.5)
    ax.axhline(0, color='0.8', ls=':', lw=0.8, zorder=0)
    ax.axvline(0, color='0.8', ls=':', lw=0.8, zorder=0)
    ax.set_xlabel('x2 (chordwise) [m]'); ax.set_ylabel('x3 (flapwise) [m]')
    ax.set_aspect('equal')
    ax.text(0.01, 0.02, 'IEA %s  (mid-surface, ref-axis origin)' % tag, transform=ax.transAxes,
            fontsize=9, color='0.35')

    # legend OUTSIDE the axes (layups + e2/e3 arrow keys)
    handles = [Line2D([0], [0], color=cols[si], lw=2, label='layup_%d' % si) for si in range(nsets)]
    handles += [Line2D([0], [0], color='blue', lw=1.2, marker='>', label='e2 (in-plane ply)'),
                Line2D([0], [0], color='black', lw=1.2, marker='>', label='e3 (wall normal)')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8.5,
              framealpha=0.95, borderaxespad=0.)
    fig.savefig(os.path.join(OUT, 'iea_%s_orient.png' % tag), dpi=140, bbox_inches='tight')
    plt.close(fig)
    return tag


sel = sys.argv[1] if len(sys.argv) > 1 else None
files = sorted(glob.glob(os.path.join(YDIR, 'iea_r*_shell.yaml')))
if sel:
    files = [f for f in files if sel in os.path.basename(f)]
for f in files:
    print('wrote orientation_png/iea_%s_orient.png' % plot_one(f))
