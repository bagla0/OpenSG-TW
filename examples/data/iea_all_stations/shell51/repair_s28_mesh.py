'''Repair the s28 (or any) 2d solid yaml: WELD coincident nodes (the mesher failed to merge them),
drop the resulting degenerate (zero-area / repeated-vertex) triangles, renumber nodes,
remap element labels in `sets` and the parallel `elementOrientations`. Writes back in the exact
original yaml format (nodes/elements as space-separated single-string flow lists, orientations
as comma float lists). Backs up the original to *.yaml.orig.

Usage: python repair_s28_mesh.py <tag>   e.g.  python repair_s28_mesh.py s28
'''
import os
import sys
import shutil
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else 's28'
PATH = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % TAG)
ROUND = 6          # coordinate rounding for coincidence (1e-6 m)
AREA_TOL = 1e-9    # delete triangles with area below this after welding


def toks(r):
    if isinstance(r, str):
        return r.split()
    if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
        return r[0].split()
    return [str(x) for x in r]


d = yaml.safe_load(open(PATH))
nodes = np.array([[float(toks(r)[0]), float(toks(r)[1]), float(toks(r)[2])] for r in d['nodes']])
elems = [[int(x) for x in toks(r)] for r in d['elements']]        # 1-indexed
orient = d['elementOrientations']
sets = d['sets']
materials = d['materials']
nn0, ne0 = len(nodes), len(elems)

# ---- 1. weld coincident nodes: map every node to the canonical (lowest) index of its coord group
canon = {}
node_map = np.arange(nn0)            # old 0-index -> canonical 0-index
for i in range(nn0):
    k = (round(nodes[i, 0], ROUND), round(nodes[i, 1], ROUND), round(nodes[i, 2], ROUND))
    if k in canon:
        node_map[i] = canon[k]
    else:
        canon[k] = i
n_welded = int((node_map != np.arange(nn0)).sum())

# ---- 2. apply weld to element connectivity (elems are 1-indexed) & find degenerate tris
def tri_area(tri0):  # tri0 = 0-indexed canonical node ids
    p = nodes[tri0]
    return 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))

keep = []
for e in elems:
    tri0 = [node_map[i - 1] for i in e]            # welded, 0-indexed
    if len(set(tri0)) < 3:
        keep.append(False)                          # collapsed (repeated vertex)
    elif tri_area(tri0) < AREA_TOL:
        keep.append(False)                          # zero-area
    else:
        keep.append(True)
keep = np.array(keep)
n_dropped = int((~keep).sum())

# ---- 3. renumber nodes: only nodes still referenced by a kept element survive
used = set()
for e, k in zip(elems, keep):
    if k:
        for i in e:
            used.add(node_map[i - 1])
used = sorted(used)
old2new = {old: j + 1 for j, old in enumerate(used)}   # canonical 0-idx -> new 1-idx
new_nodes = nodes[used]

# ---- 4. build new elements + orientations + old-label->new-label map (1-indexed)
new_elems = []
new_orient = []
label_map = {}     # old 1-idx elem label -> new 1-idx elem label
new_lab = 0
for old_lab, (e, k) in enumerate(zip(elems, keep), start=1):
    if not k:
        continue
    new_lab += 1
    label_map[old_lab] = new_lab
    new_elems.append([old2new[node_map[i - 1]] for i in e])
    new_orient.append(orient[old_lab - 1])

# ---- 5. remap sets element labels
new_sets = {'element': []}
for grp in sets['element']:
    new_labels = [label_map[l] for l in grp['labels'] if l in label_map]
    new_sets['element'].append({'name': grp['name'], 'labels': new_labels})

print('%s: nodes %d -> %d (welded %d)   elements %d -> %d (dropped %d)' % (
    TAG, nn0, len(new_nodes), n_welded, ne0, len(new_elems), n_dropped))

# ---- 6. write yaml in the ORIGINAL format
shutil.copy(PATH, PATH + '.orig')
lines = []
lines.append('nodes:')
for nd in new_nodes:
    lines.append('- [%.8f %.8f %.8f]' % (nd[0], nd[1], nd[2]))
lines.append('elements:')
for e in new_elems:
    lines.append('- [%d %d %d]' % (e[0], e[1], e[2]))
lines.append('sets:')
lines.append('  element:')
for grp in new_sets['element']:
    lines.append('  - name: %s' % grp['name'])
    lines.append('    labels: [%s]' % ', '.join(str(x) for x in grp['labels']))
lines.append('elementOrientations:')
for o in new_orient:
    lines.append('- [%s]' % ', '.join(repr(float(x)) for x in o))
lines.append('materials:')
for m in materials:
    lines.append('- name: %s' % m['name'])
    lines.append('  E: [%s]' % ', '.join(repr(float(x)) for x in m['E']))
    lines.append('  G: [%s]' % ', '.join(repr(float(x)) for x in m['G']))
    lines.append('  nu: [%s]' % ', '.join(repr(float(x)) for x in m['nu']))
    lines.append('  rho: %s' % repr(float(m['rho'])))
open(PATH, 'w').write('\n'.join(lines) + '\n')

# ---- 7. verify it re-parses and re-check quality
d2 = yaml.safe_load(open(PATH))
n2 = np.array([[float(toks(r)[0]), float(toks(r)[1])] for r in d2['nodes']])
e2 = [[int(x) for x in toks(r)] for r in d2['elements']]
seen = {}
nc = 0
for i, nd in enumerate(n2):
    k = (round(nd[0], 6), round(nd[1], 6))
    if k in seen:
        nc += 1
    seen[k] = i
zero = 0
for e in e2:
    p = n2[[i - 1 for i in e]]
    A = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
    if A < 1e-10:
        zero += 1
print('post-repair: reparsed nn=%d ne=%d  coincident=%d  zero-area=%d  (orient len=%d)' % (
    len(n2), len(e2), nc, zero, len(d2['elementOrientations'])))
assert len(d2['elementOrientations']) == len(e2), 'orientation/element length mismatch!'
