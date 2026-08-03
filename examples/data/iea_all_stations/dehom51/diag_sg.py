'''diagnose sg00/sg02 (negative Jacobian?) + inventory clean pyNuMAD-quad yamls for 00,02,50.'''
import os, glob, numpy as np, yaml
ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
SGV = os.path.join(ROOT, 'shell51/sg_v201')


def parse_sg(path):
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    hdr = None
    for i, l in enumerate(lines):
        t = l.split()
        if len(t) == 3 and all(x.lstrip('-').isdigit() for x in t) and int(t[0]) > 1000:
            hdr = i; break
    nn, ne = int(lines[hdr].split()[0]), int(lines[hdr].split()[1])
    xy = np.array([[float(lines[hdr + 1 + k].split()[1]), float(lines[hdr + 1 + k].split()[2])]
                   for k in range(nn)])
    conn = []
    for k in range(ne):
        t = [int(x) for x in lines[hdr + 1 + nn + k].split()]
        conn.append([n for n in t[1:] if n != 0])            # drop id + zero pad
    return xy, conn


def signed_areas(xy, conn):
    a = []
    for c in conn:
        p = xy[[n - 1 for n in c]]
        s = 0.0
        for i in range(len(p)):
            x1, y1 = p[i]; x2, y2 = p[(i + 1) % len(p)]
            s += x1 * y2 - x2 * y1
        a.append(0.5 * s)
    return np.array(a)


print('=== sg_v201 mesh Jacobian check (negative/zero signed area = inverted element) ===')
for s in ('iea_s00', 'iea_s02', 'iea_s01', 'iea_s50'):
    p = os.path.join(SGV, s + '.sg')
    if not os.path.exists(p):
        print('  %-8s : NO .sg' % s); continue
    xy, conn = parse_sg(p)
    ar = signed_areas(xy, conn)
    nneg = int((ar < 0).sum()); nzero = int((np.abs(ar) < 1e-14).sum())
    ncnt = sorted(set(len(c) for c in conn))
    print('  %-8s : nodes=%d elems=%d types=%s  neg=%d zero=%d  |area|min=%.2e'
          % (s, len(xy), len(conn), ncnt, nneg, nzero, np.abs(ar).min()))

print('\n=== available clean pyNuMAD-quad yamls for 00,02,50 ===')
for st in ('s00', 's02', 's50'):
    for d in ('pynumad_quad', '2d_hybrid', 'fallback_yaml', 'robust_yaml'):
        for f in glob.glob(os.path.join(ROOT, 'shell51', d, 'iea_%s_solid*.yaml' % st)):
            try:
                dd = yaml.safe_load(open(f))
                els = dd['elements']
                row = lambda x: [float(v) for v in (x[0].split() if isinstance(x, list) else str(x).split())]
                nd = np.array([row(n)[:2] for n in dd['nodes']])
                elens = sorted(set(len(row(e)) for e in els))
                print('  %-40s nodes=%d elems=%d elens=%s bbox_x=[%.3f,%.3f] y=[%.3f,%.3f]'
                      % (os.path.relpath(f, ROOT), len(nd), len(els), elens,
                         nd[:, 0].min(), nd[:, 0].max(), nd[:, 1].min(), nd[:, 1].max()))
            except Exception as e:
                print('  %-40s (err %s)' % (os.path.relpath(f, ROOT), e))
