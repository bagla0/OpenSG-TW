'''Triangle-mesh quality for the 2d solid yaml: coincident nodes, zero-area / sliver triangles,
degenerate (repeated-vertex) triangles. Compare s27/s28/s29.'''
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    p = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % tag)
    d = yaml.safe_load(open(p))
    def toks(r):
        if isinstance(r, str):
            return r.split()
        if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
            return r[0].split()
        return [str(x) for x in r]
    nodes = np.array([[float(toks(r)[0]), float(toks(r)[1])] for r in d['nodes']])
    elems = [[int(x) for x in toks(r)] for r in d['elements']]
    return nodes, elems


def tri_area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


for tag in ('s27', 's28', 's29'):
    nodes, elems = load(tag)
    # detect 0- vs 1-indexing
    allidx = [i for e in elems for i in e]
    base = 1 if min(allidx) >= 1 and max(allidx) >= len(nodes) else (1 if min(allidx) == 1 else 0)
    # robust: if max index == len(nodes) then 1-indexed
    base = 1 if max(allidx) >= len(nodes) else 0

    # coincident nodes (round to 1e-6)
    seen = {}
    ncoinc = 0
    coinc_pairs = []
    for i, nd in enumerate(nodes):
        k = (round(nd[0], 6), round(nd[1], 6))
        if k in seen:
            ncoinc += 1
            coinc_pairs.append((seen[k], i))
        else:
            seen[k] = i

    # triangle metrics
    areas = []
    degen_repeat = 0  # triangle with a repeated vertex index
    slivers = 0
    coinc_node_set = set()
    for a, b in coinc_pairs:
        coinc_node_set.add(a)
        coinc_node_set.add(b)
    tris_using_coinc = 0
    for e in elems:
        idx = [i - base for i in e]
        if len(set(idx)) < 3:
            degen_repeat += 1
        p = nodes[idx]
        A = tri_area(p[0], p[1], p[2])
        areas.append(A)
        peri = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[1]) + np.linalg.norm(p[0] - p[2]))
        if peri > 0 and (4 * np.sqrt(3) * A) / (peri ** 2) < 0.02:  # quality << 1 -> sliver
            slivers += 1
        if any(i in coinc_node_set for i in idx):
            tris_using_coinc += 1
    areas = np.array(areas)
    print('=== %s (base=%d, nn=%d, ne=%d) ===' % (tag, base, len(nodes), len(elems)))
    print('  coincident node pairs: %d   (nodes involved: %d)' % (len(coinc_pairs), len(coinc_node_set)))
    print('  zero-area tris (<1e-10): %d   min area=%.3e' % (int((areas < 1e-10).sum()), areas.min()))
    print('  sliver tris (quality<0.02): %d' % slivers)
    print('  repeated-vertex tris: %d' % degen_repeat)
    print('  tris touching a coincident node: %d' % tris_using_coinc)
    if coinc_pairs[:5]:
        print('  sample coincident pairs (idx):', coinc_pairs[:5])
