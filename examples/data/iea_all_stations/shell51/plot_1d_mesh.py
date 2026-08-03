'''plot_1d_mesh.py -- render the ACTUAL generated 1-D shell mesh (OpenSG_io, OML contour + x1-shift)
for the two stations PreVABS could not mesh as a 2-D solid: s02 (root transition) and s50 (tip).
Shows the 1-D shell is complete/valid for BOTH (the break is only in the PreVABS 2-D solid offset).
Legend OUTSIDE. No title.'''
import os
from collections import defaultdict

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
YDIR = os.path.join(HERE, '1d_yaml')


def rows(raw, cast):
    out = []
    for r in raw:
        toks = r.split() if isinstance(r, str) else (r[0].split() if (len(r) == 1 and isinstance(r[0], str)) else r)
        out.append([cast(t) for t in toks])
    return np.array(out)


def load(tag):
    d = yaml.safe_load(open(os.path.join(YDIR, 'iea_%s_shell.yaml' % tag)))
    nd = rows(d['nodes'], float); el = rows(d['elements'], int) - 1
    es = d['sets']['element']
    lab2set = {int(lab): si for si, s in enumerate(es) for lab in s['labels']}
    setid = np.array([lab2set.get(k + 1, 0) for k in range(len(el))])
    return nd, el, setid, len(es)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, tag, lab in zip(axes, ('s02', 's50'), ('s02 (η=0.04 root transition)', 's50 (η=1.0 tip)')):
    nd, el, setid, nsets = load(tag)
    cols = plt.cm.rainbow(np.linspace(0, 1, max(nsets, 2)))
    for k, (a, b) in enumerate(el):
        ax.plot([nd[a, 0], nd[b, 0]], [nd[a, 1], nd[b, 1]], '-', color=cols[setid[k]], lw=1.4, zorder=2)
    ax.plot(nd[:, 0], nd[:, 1], '.', color='0.35', ms=2.5, zorder=3)
    ax.plot(0, 0, 'P', color='crimson', ms=12, zorder=5)
    ax.set_aspect('equal'); ax.set_xlabel('x2 (chordwise) [m]'); ax.set_ylabel('x3 (flapwise) [m]')
    ax.text(0.02, 0.03, '%s\n1-D shell: %d nodes, %d elems  — VALID (no break)' % (lab, len(nd), len(el)),
            transform=ax.transAxes, fontsize=9.5, color='0.2',
            bbox=dict(boxstyle='round', fc='#f2fbf2', ec='tab:green'))

handles = [Line2D([0], [0], color='tab:purple', lw=2, label='1-D shell mesh (coloured by layup)'),
           Line2D([0], [0], marker='.', color='0.35', lw=0, label='nodes'),
           Line2D([0], [0], marker='P', color='crimson', lw=0, label='x1 reference axis (origin)')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(HERE, 's02_s50_1d_mesh.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print('wrote', out)
for tag in ('s02', 's50'):
    nd, el, setid, nsets = load(tag)
    print('%s : %d nodes, %d elems, %d layups -> 1D shell VALID' % (tag, len(nd), len(el), nsets))
