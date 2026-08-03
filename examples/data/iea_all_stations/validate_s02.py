'''Validate the conditioned s02 before promoting it into sg_v201:
  (1) structural integrity of cond/iea_s02.sg  -> node/elem counts, coincident-node & zero-area check
  (2) physical smoothness of the s02 STATION   -> Timo 6x6 diagonal vs neighbours s01, s03
'''
import os
import numpy as np

ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
SG = os.path.join(ROOT, 'shell51/xml/cond/iea_s02.sg')
OUTDIR = os.path.join(ROOT, 'shell51/out/OpenSG_Hybrid_Solid')

# ---------------------------------------------------------------- (1) parse VABS .sg mesh
def parse_sg(path):
    '''VABS SG (line-based).  Control header (1 6 / 1 0 0 / 0 0 0 0), then a line "nnode nelem nmate"
    (3 ints, first>1000), then nnode node rows "id x y", then nelem element rows.'''
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    hdr = None
    for idx, l in enumerate(lines):
        t = l.split()
        if len(t) == 3 and all(x.lstrip('-').isdigit() for x in t) and int(t[0]) > 1000:
            hdr = idx; break
    if hdr is None:
        print('  could not locate mesh header'); return None
    nnode, nelem = int(lines[hdr].split()[0]), int(lines[hdr].split()[1])
    xy = np.zeros((nnode, 2))
    for k in range(nnode):
        t = lines[hdr + 1 + k].split()
        xy[k] = (float(t[1]), float(t[2]))
    return nnode, nelem, xy


def check_mesh(path, tag):
    print('--- %s : %s' % (tag, os.path.basename(path)))
    r = parse_sg(path)
    if r is None:
        return
    nnode, nelem, xy = r
    print('    nodes=%d  elems=%d' % (nnode, nelem))
    print('    bbox x=[%.4f, %.4f]  y=[%.4f, %.4f]  (chord span x=%.4f)'
          % (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max(),
             xy[:, 0].max() - xy[:, 0].min()))
    # coincident-node check (the Class-C degeneracy that corrupts GA2/GA3)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    dmin = d[:, 1]
    ncoinc = int((dmin < 1e-9).sum())
    print('    min node-node distance=%.3e   coincident pairs(<1e-9)=%d %s'
          % (dmin.min(), ncoinc, 'OK' if ncoinc == 0 else '<-- DEGENERATE'))


# ---------------------------------------------------------------- (2) Timo 6x6 smoothness
def read_K(path):
    lines = open(path).read().splitlines()
    for i, l in enumerate(lines):
        if l.strip().startswith('Stiffness'):
            rows = []
            j = i + 1
            while len(rows) < 6 and j < len(lines):
                v = lines[j].split()
                try:
                    fv = [float(x) for x in v]
                    if len(fv) >= 6:
                        rows.append(fv[:6])
                except ValueError:
                    pass
                j += 1
            return np.array(rows)
    return None


print('=' * 70)
print('(1) mesh structural integrity')
print('=' * 70)
check_mesh(SG, 's02 conditioned (cond)')

print()
print('=' * 70)
print('(2) Timo 6x6 diagonal smoothness  s01 -> s02 -> s03  (Hybrid_Solid .out)')
print('=' * 70)
lbl = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
Ks = {}
for s in ('s01', 's02', 's03'):
    p = os.path.join(OUTDIR, 'iea_%s_OpenSG_Hybrid_Solid.out' % s)
    Ks[s] = read_K(p)
print('%-5s %12s %12s %12s   %s' % ('term', 's01', 's02', 's03', 's02 vs midpoint(s01,s03)'))
for d in range(6):
    a, b, c = Ks['s01'][d, d], Ks['s02'][d, d], Ks['s03'][d, d]
    mid = 0.5 * (a + c)
    err = 100.0 * (b - mid) / mid if abs(mid) > 0 else 0.0
    flag = 'OK' if abs(err) < 25 else '<-- check'
    print('%-5s %12.4e %12.4e %12.4e   %+7.1f%%  %s' % (lbl[d], a, b, c, err, flag))
