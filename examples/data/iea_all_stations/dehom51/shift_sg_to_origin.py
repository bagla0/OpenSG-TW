'''shift every sg_v201 .sg to the (0,0) reference axis (x -= section_offset_y(eta), y unchanged),
so VABS runs at the same origin as OpenSG.  Backs up the LE originals to sg_v201_LE, overwrites
sg_v201 + dehom_iea/sg_v201.  Pure meshes (tri for 49, quad for s02/s50) -- no hybrid.'''
import os, shutil
import numpy as np
import yaml
ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
SGV = os.path.join(ROOT, 'shell51/sg_v201')
BAK = os.path.join(ROOT, 'shell51/sg_v201_LE')
DEH = os.path.join(ROOT, 'dehom_iea/sg_v201')
os.makedirs(BAK, exist_ok=True)

d = yaml.safe_load(open(os.path.join(ROOT, 'IEA-22-280-RWT.yaml')))
so = d['components']['blade']['outer_shape']['section_offset_y']
G, V = np.array(so['grid'], float), np.array(so['values'], float)


def soffs(e):
    return float(np.interp(e, G, V))


def shift_sg(path, dx):
    lines = open(path).read().splitlines()
    nb = [i for i, l in enumerate(lines) if l.strip()]                 # non-blank line indices
    hidx = nb[3]                                                       # 4th non-blank line = nnode nelem nmate
    nn = int(lines[hidx].split()[0])
    xs = []
    for k in range(hidx + 1, len(lines)):                             # advance through the node block
        if not lines[k].strip():
            continue
        t = lines[k].split()
        if len(t) < 3:
            break
        nid, x, y = t[0], float(t[1]) - dx, float(t[2])
        lines[k] = '%8s%20.9e%20.9e' % (nid, x, y)
        xs.append(x)
        if len(xs) == nn:
            break
    open(path, 'w').write('\n'.join(lines) + '\n')
    return nn, min(xs), max(xs)


print('shift each .sg to (0,0) reference axis:')
for i in range(51):
    p = os.path.join(SGV, 'iea_s%02d.sg' % i)
    if not os.path.exists(p):
        print('  s%02d MISSING' % i); continue
    dx = soffs(i / 50.0)
    bakp = os.path.join(BAK, 'iea_s%02d.sg' % i)
    if os.path.exists(bakp):
        shutil.copy(bakp, p)                                          # restore LE (idempotent re-run)
    else:
        shutil.copy(p, bakp)                                          # first time: backup the LE original
    nn, xmin, xmax = shift_sg(p, dx)
    if i % 10 == 0 or i in (2, 50):
        print('  s%02d dx=%.4f -> x[%.3f, %.3f] (crosses 0: %s)  n=%d'
              % (i, dx, xmin, xmax, xmin < 0 < xmax, nn))

# mirror into the dehom_iea VABS folder
for i in range(51):
    src = os.path.join(SGV, 'iea_s%02d.sg' % i)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DEH, 'iea_s%02d.sg' % i))
    m = src + '.mat'
    if os.path.exists(m):
        shutil.copy(m, os.path.join(DEH, 'iea_s%02d.sg.mat' % i))
print('done: sg_v201 + dehom_iea/sg_v201 now at (0,0); LE originals backed up in sg_v201_LE')
