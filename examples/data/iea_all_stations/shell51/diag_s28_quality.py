'''Look for the s28 solid mesh defect that doubles GA2/GA3.
Per-element min edge / aspect / signed area (2D quad section) + material-set + orientation stats,
comparing s28 to s27 and s29.'''
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    p = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % tag)
    d = yaml.safe_load(open(p))
    nodes = []
    for r in d['nodes']:
        t = (r if isinstance(r, str) else r[0]).split()
        nodes.append([float(t[0]), float(t[1])])
    nodes = np.array(nodes)
    elems = []
    for r in d['elements']:
        if isinstance(r, str):
            elems.append([int(x) for x in r.split()])
        else:
            elems.append([int(x) for x in r])
    return d, nodes, elems


def quad_area(pts):
    # shoelace for polygon (first 4 nodes = corners for quad/hex-section)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


for tag in ('s27', 's28', 's29'):
    d, nodes, elems = load(tag)
    areas = []
    minedge = []
    aspect = []
    ncorner = []
    for e in elems:
        idx = [i - 1 if max(e) >= len(nodes) else i for i in e]  # 0/1-index guess
        try:
            pts = nodes[idx[:4]] if len(idx) >= 4 else nodes[idx]
        except IndexError:
            idx = [i - 1 for i in e]
            pts = nodes[idx[:4]]
        ncorner.append(len(e))
        a = quad_area(pts)
        areas.append(a)
        # edges of the corner polygon
        eln = [np.linalg.norm(pts[(j + 1) % len(pts)] - pts[j]) for j in range(len(pts))]
        eln = [x for x in eln if x > 0]
        if eln:
            minedge.append(min(eln))
            aspect.append(max(eln) / min(eln))
    areas = np.array(areas)
    minedge = np.array(minedge)
    aspect = np.array(aspect)
    # materials / sets
    mats = d.get('materials', [])
    nmat = len(mats) if hasattr(mats, '__len__') else mats
    sets = d.get('sets', [])
    nsets = len(sets) if hasattr(sets, '__len__') else sets
    eo = d.get('elementOrientations', [])
    neo = len(eo) if hasattr(eo, '__len__') else eo
    print('=== %s ===' % tag)
    print('  elems=%d  corner-counts=%s' % (len(elems), sorted(set(ncorner))))
    print('  area: min=%.3e max=%.3e  #zero(<1e-10)=%d' % (areas.min(), areas.max(), int((areas < 1e-10).sum())))
    print('  minedge: min=%.3e  #tiny(<1e-4)=%d' % (minedge.min(), int((minedge < 1e-4).sum())))
    print('  aspect: max=%.1f  #high(>50)=%d' % (aspect.max(), int((aspect > 50).sum())))
    print('  nmat=%d nsets=%d nElemOrient=%d' % (nmat, nsets, neo))
    # duplicate elements?
    keyset = set()
    dup = 0
    for e in elems:
        k = tuple(sorted(e))
        if k in keyset:
            dup += 1
        keyset.add(k)
    print('  duplicate elements: %d' % dup)
