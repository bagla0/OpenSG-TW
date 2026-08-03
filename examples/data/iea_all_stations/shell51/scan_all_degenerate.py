'''Scan every 2d solid yaml for coincident nodes / zero-area triangles (the s28 defect class).
Report which stations need the weld repair.'''
import os
import glob
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def toks(r):
    if isinstance(r, str):
        return r.split()
    if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
        return r[0].split()
    return [str(x) for x in r]


bad = []
for p in sorted(glob.glob(os.path.join(HERE, '2d_yaml', '*_solid.yaml'))):
    tag = os.path.basename(p).replace('iea_', '').replace('_solid.yaml', '')
    d = yaml.safe_load(open(p))
    nodes = np.array([[float(toks(r)[0]), float(toks(r)[1])] for r in d['nodes']])
    elems = [[int(x) for x in toks(r)] for r in d['elements']]
    seen = {}
    nc = 0
    for i, nd in enumerate(nodes):
        k = (round(nd[0], 6), round(nd[1], 6))
        if k in seen:
            nc += 1
        seen[k] = i
    zero = 0
    for e in elems:
        pp = nodes[[i - 1 for i in e]]
        A = 0.5 * abs((pp[1, 0] - pp[0, 0]) * (pp[2, 1] - pp[0, 1]) - (pp[2, 0] - pp[0, 0]) * (pp[1, 1] - pp[0, 1]))
        if A < 1e-10:
            zero += 1
    flag = ' <-- REPAIR' if (nc or zero) else ''
    if nc or zero:
        bad.append(tag)
    print('%-6s nn=%-6d ne=%-6d coincident=%d zero-area=%d%s' % (tag, len(nodes), len(elems), nc, zero, flag))

print('\nstations needing repair: %s' % (bad if bad else 'NONE'))
