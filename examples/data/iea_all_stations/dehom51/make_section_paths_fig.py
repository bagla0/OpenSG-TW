'''make_section_paths_fig.py -- the r=0.2 2-D cross-section (solid mesh colored by
material) with the two dehom paths overlaid: the circumferential LP-OML loop and the
spar-cap through-thickness column, the latter with explicit START (outer surface) and
END (inner surface) markers.  Portable paths; output out/r020_section_paths.png.'''
import json
import os
from collections import defaultdict

import numpy as np
import yaml
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, '..'))
SOL = os.path.join(IEA, 'shell51', '2d_yaml', 'iea_s10_solid.yaml')
CO = os.path.join(HERE, 'coords')


def frow(v):
    if isinstance(v, str):
        return [float(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [float(x) for x in v[0].split()]
    return [float(x) for x in v]


def irow(v):
    if isinstance(v, str):
        return [int(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [int(x) for x in v[0].split()]
    return [int(x) for x in v]


y = yaml.safe_load(open(SOL))
xy = np.array([frow(n)[:2] for n in y['nodes']])
conn = [irow(e) for e in y['elements']]
base = min(min(c) for c in conn)
conn = [[n - base for n in c] for c in conn]
ne = len(conn)
matnames = [m['name'] for m in y['materials']]
E1 = np.array([frow(m['E'])[0] for m in y['materials']])
emat = np.zeros(ne, dtype=int)
setlist = y['sets']['element']
lab_min = min(min(s['labels']) for s in setlist)
for s in setlist:
    mi = matnames.index(s['name']) if s['name'] in matnames else 0
    for lab in s['labels']:
        emat[lab - lab_min] = mi
mnf = os.path.join(CO, 'material_names.json')
mnames = json.load(open(mnf)) if os.path.exists(mnf) else {}
realname = lambda mi: mnames.get(matnames[mi], matnames[mi])
carbon = int(np.argmax(E1))

circ = np.loadtxt(os.path.join(CO, 'iea_s10.circumferential.coords'))[:, :2]
col = np.loadtxt(os.path.join(CO, 'iea_s10.lp_sparcap_left_thickness.coords'))[:, :2]

cmap = plt.cm.tab10
fig, ax = plt.subplots(figsize=(13, 5))
ax.add_collection(PolyCollection([xy[c] for c in conn],
                                 facecolors=[cmap(emat[k] % 10) for k in range(ne)],
                                 edgecolors='none'))
handles = [Patch(facecolor=cmap(mi % 10),
                 label='%s%s' % (realname(mi), '  (carbon)' if mi == carbon else ''))
           for mi in range(len(matnames))]

ax.plot(circ[:, 0], circ[:, 1], '-', color='k', lw=2.6,
        label='circumferential path (LE $\\rightarrow$ TE)')
ax.plot(col[:, 0], col[:, 1], '-', color='cyan', lw=3.4,
        label='through-thickness path (outer $\\rightarrow$ inner)')
# ONE direction arrow per path.
# circumferential: tangent arrow at ~40% of the path, pointing toward the TE
i0 = int(0.40 * len(circ))
d = circ[min(i0 + 3, len(circ) - 1)] - circ[i0 - 3]
d = d / np.linalg.norm(d)
ax.annotate('', xy=circ[i0] + 0.45 * d, xytext=circ[i0],
            arrowprops=dict(arrowstyle='-|>', color='k', lw=2.6,
                            mutation_scale=28))
# through-thickness: one arrow from outside the outer surface pointing inward
dt = col[-1] - col[0]
dt = dt / np.linalg.norm(dt)
ax.annotate('', xy=col[-1] + 0.16 * dt, xytext=col[0] - 0.30 * dt,
            arrowprops=dict(arrowstyle='-|>', color='cyan', lw=3.0,
                            mutation_scale=30))
ax.plot([0], [0], '+', color='k', mew=2.0, ms=14)
ax.set_aspect('equal')
ax.autoscale()
ax.axis('off')
lg1 = ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.add_artist(lg1)
ax.legend(handles=handles, loc='upper right', fontsize=10, title='materials',
          framealpha=0.95)
fig.tight_layout()
out = os.path.join(HERE, 'out', 'r020_section_paths.png')
fig.savefig(out, dpi=160, bbox_inches='tight')
print('wrote', out)
