'''build_s10_oml_from_center.py -- OML 1-D yaml for iea_s10 derived FROM the stored
center (mid-surface) yaml, NOT by re-parsing the XML: the current parse drops the
32.6 mm layup region (5 sections vs the correct 6; EA -6.8% vs .K), while the stored
center yaml reproduces the VABS .K within the expected RM gap (EA -1.45%).

Transform: skin contour nodes move OUTWARD by t_e/2 along the element wall normal
(node value = average over adjacent skin elements); web chains keep their line, but
junction nodes move with the skin (the web extends to the OML skin line, as an
XML-built OML mesh would).  Sections/sets/orientations/materials are untouched;
`reference` becomes "oml".  Overwrites shell51/1d_yaml_oml/iea_s10_shell.yaml.
'''
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, '..'))
CEN = os.path.join(IEA, 'shell51', '1d_yaml', 'iea_s10_shell.yaml')
OUT = os.path.join(IEA, 'shell51', '1d_yaml_oml', 'iea_s10_shell.yaml')


def rows(seq):
    out = []
    for r in seq:
        if isinstance(r, str):
            out.append([float(x) for x in r.split()])
        elif isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
            out.append([float(x) for x in r[0].split()])
        else:
            out.append([float(x) for x in r])
    return np.array(out)


d = yaml.safe_load(open(CEN))
rx = rows(d['nodes'])[:, :2]
cells = rows(d['elements']).astype(int)
if cells.min() == 1:
    cells = cells - 1
ori = rows(d['elementOrientations'])
e3 = ori[:, 6:9]

secs = d['sections']
thick = {s['elementSet']: sum(float(p[1]) for p in s['layup']) for s in secs}
elem_t = np.zeros(len(cells))
for g in d['sets']['element']:
    for lab in g['labels']:
        elem_t[int(lab) - 1] = thick[g['name']]

# web detection: straight near-vertical chain between junction (deg>=3) nodes
deg = np.zeros(len(rx), dtype=int)
for a, b in cells:
    deg[a] += 1
    deg[b] += 1
junc = set(np.where(deg >= 3)[0])
adj = {}
for e, (a, b) in enumerate(cells):
    adj.setdefault(a, []).append((b, e))
    adj.setdefault(b, []).append((a, e))
is_web = np.zeros(len(cells), dtype=bool)
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
print('web elements:', int(is_web.sum()), 'of', len(cells),
      '| junctions:', len(junc))

# outward in-plane normal per element (away from centroid)
cen = rx.mean(0)
n2d = e3[:, :2].copy()
nrm = np.linalg.norm(n2d, axis=1)
bad = nrm < 1e-8
# fallback normal from the tangent if e3 has no in-plane part
tang = rx[cells[:, 1]] - rx[cells[:, 0]]
tang /= np.linalg.norm(tang, axis=1, keepdims=True).clip(1e-30)
n2d[bad] = np.column_stack([tang[bad, 1], -tang[bad, 0]])
n2d /= np.linalg.norm(n2d, axis=1, keepdims=True).clip(1e-30)
mid = 0.5 * (rx[cells[:, 0]] + rx[cells[:, 1]])
sgn = np.sign(((mid - cen) * n2d).sum(1))
sgn[sgn == 0] = 1.0
n2d *= sgn[:, None]

off = np.zeros_like(rx)
cnt = np.zeros(len(rx))
for e, (a, b) in enumerate(cells):
    if is_web[e]:
        continue
    for nd in (a, b):
        off[nd] += 0.5 * elem_t[e] * n2d[e]
        cnt[nd] += 1
m = cnt > 0
off[m] /= cnt[m][:, None]
rx_oml = rx + off
print('offset nodes:', int(m.sum()), 'of', len(rx),
      '| mean |offset| %.1f mm' % (np.linalg.norm(off[m], axis=1).mean() * 1e3))

# write: same yaml, new nodes (SAME row format as the original), reference: oml
orig = rows(d['nodes'])
first = d['nodes'][0]
as_str = isinstance(first, str) or (isinstance(first, list) and len(first) == 1
                                    and isinstance(first[0], str))
nodes_out = []
for i in range(len(rx_oml)):
    z = orig[i, 2] if orig.shape[1] > 2 else 0.0
    if as_str:
        s = '%0.10f %0.10f %0.1f' % (rx_oml[i, 0], rx_oml[i, 1], z)
        nodes_out.append([s] if isinstance(first, list) else s)
    else:
        nodes_out.append([float(rx_oml[i, 0]), float(rx_oml[i, 1]), float(z)])
d['nodes'] = nodes_out
d['reference'] = 'oml'
with open(OUT, 'w') as f:
    yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False, width=1000)
print('wrote', OUT)
