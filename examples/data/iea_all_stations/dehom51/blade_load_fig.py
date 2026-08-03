'''blade_load_fig.py -- the full IEA-22 blade lofted from the 51 station contours,
colored by layup REGION (section index; color = region, layup may vary along span),
with arrows showing the applied 1.5 kPa flapwise surface traction and an x1-x2-x3
triad.  No axes.  Skin loops only (webs excluded), resampled to a common
parameterization so stations with different node counts loft cleanly.
Output: out/blade_load.png
'''
import glob
import os

import numpy as np
import yaml
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, '..'))
L_BLADE = 137.0
NS = 200                      # common resampling points per station loop


def rows(seq, typ=float):
    out = []
    for r in seq:
        if isinstance(r, str):
            out.append([typ(x) for x in r.split()])
        elif isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
            out.append([typ(x) for x in r[0].split()])
        else:
            out.append([typ(x) for x in r])
    return out


def skin_loop(path):
    d = yaml.safe_load(open(path))
    rx = np.array(rows(d['nodes']))[:, :2]
    cells = np.array(rows(d['elements'], int), dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    # region per element
    secname = {s['elementSet']: k for k, s in enumerate(d['sections'])}
    reg = np.zeros(len(cells), dtype=int)
    for g in d['sets']['element']:
        for lab in g['labels']:
            reg[int(lab) - 1] = secname[g['name']]
    # web removal by topology (junction-to-junction near-vertical straight chains)
    deg = np.zeros(len(rx), int)
    for a, b in cells:
        deg[a] += 1
        deg[b] += 1
    junc = set(np.where(deg >= 3)[0])
    adj = {}
    for e, (a, b) in enumerate(cells):
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))
    is_web = np.zeros(len(cells), bool)
    seen = set()
    for j in junc:
        for (nxt, e0) in adj[j]:
            if e0 in seen:
                continue
            chain, prev, cur = [e0], j, nxt
            seen.add(e0)
            while cur not in junc and deg[cur] == 2:
                (n1, e1), (n2, e2) = adj[cur][0], adj[cur][1]
                nn, ee = (n1, e1) if n1 != prev else (n2, e2)
                if ee in seen:
                    break
                chain.append(ee)
                seen.add(ee)
                prev, cur = cur, nn
            if cur in junc:
                arc = sum(np.linalg.norm(rx[cells[c][1]] - rx[cells[c][0]]) for c in chain)
                cv = rx[cur] - rx[j]
                ch = np.linalg.norm(cv)
                if ch / max(arc, 1e-30) > 0.99 and abs(cv[1]) / max(ch, 1e-30) > 0.6:
                    is_web[chain] = True
    skin = np.where(~is_web)[0]
    mid = 0.5 * (rx[cells[skin, 0]] + rx[cells[skin, 1]])
    cen = rx.mean(0)
    ang = np.arctan2(mid[:, 1] - cen[1], mid[:, 0] - cen[0])
    order = skin[np.argsort(ang)]
    pts = 0.5 * (rx[cells[order, 0]] + rx[cells[order, 1]])
    regs = reg[order]
    # anchor at the TE (max x) and resample by cumulative arc fraction
    i0 = int(np.argmax(pts[:, 0]))
    pts = np.roll(pts, -i0, axis=0)
    regs = np.roll(regs, -i0)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    sfrac = np.r_[0.0, np.cumsum(seg)][:-1]
    sfrac /= sfrac[-1] + seg[-1]
    tq = np.linspace(0, 1, NS, endpoint=False)
    x = np.interp(tq, sfrac, pts[:, 0], period=1.0)
    y = np.interp(tq, sfrac, pts[:, 1], period=1.0)
    ridx = np.searchsorted(sfrac, tq, side='right') - 1
    return np.column_stack([x, y]), regs[np.clip(ridx, 0, len(regs) - 1)]


files = sorted(glob.glob(os.path.join(IEA, 'shell51', '1d_yaml', 'iea_s*_shell.yaml')))
P = np.zeros((len(files), NS, 2))
R = np.zeros((len(files), NS), int)
for i, f in enumerate(files):
    P[i], R[i] = skin_loop(f)
z = np.arange(len(files)) / (len(files) - 1) * L_BLADE
print('lofted %d stations x %d pts' % (len(files), NS))

cmap = plt.cm.tab10
fig = plt.figure(figsize=(14, 6.5))
ax = fig.add_subplot(111, projection='3d')
quads, cols = [], []
for i in range(len(files) - 1):
    for j in range(NS):
        j2 = (j + 1) % NS
        quads.append([(z[i], P[i, j, 0], P[i, j, 1]),
                      (z[i], P[i, j2, 0], P[i, j2, 1]),
                      (z[i + 1], P[i + 1, j2, 0], P[i + 1, j2, 1]),
                      (z[i + 1], P[i + 1, j, 0], P[i + 1, j, 1])])
        cols.append(cmap(R[i, j] % 10))
pc = Poly3DCollection(quads, facecolors=cols, edgecolors='none')
ax.add_collection3d(pc)

# 1.5 kPa flapwise traction arrows: tails on the suction (upper) surface, +x3 up
for zi in np.linspace(0.08, 0.92, 7) * L_BLADE:
    i = int(zi / L_BLADE * (len(files) - 1))
    j = int(np.argmax(P[i, :, 1]))
    x0, y0 = P[i, j]
    ax.quiver(zi, x0, y0, 0, 0, 9.0, color='crimson', lw=2.4,
              arrow_length_ratio=0.28)
ax.text(0.5 * L_BLADE, 0.0, 15.5, r'$p=1.5$ kPa flapwise traction',
        color='crimson', fontsize=13, ha='center')

# x1-x2-x3 triad at the root
t0 = np.array([-6.0, -7.5, -3.0])
for d, lab in [((10, 0, 0), '$x_1$'), ((0, 5, 0), '$x_2$'), ((0, 0, 5), '$x_3$')]:
    ax.quiver(*t0, *d, color='k', lw=2.0, arrow_length_ratio=0.18)
    ax.text(t0[0] + 1.35 * d[0], t0[1] + 1.35 * d[1], t0[2] + 1.35 * d[2], lab,
            fontsize=13, ha='center')

ax.set_xlim(-8, L_BLADE)
ax.set_ylim(-9, 9)
ax.set_zlim(-8, 17)
ax.set_box_aspect((3.4, 1.0, 0.62))
ax.view_init(elev=20, azim=-72)
ax.set_axis_off()
fig.tight_layout()
out = os.path.join(HERE, 'out', 'blade_load.png')
fig.savefig(out, dpi=170, bbox_inches='tight')
print('wrote', out)
