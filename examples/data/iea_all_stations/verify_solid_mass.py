'''verify_solid_mass.py -- compute the get_mass_solid 6x6 formula on the r0020 2-D solid mesh and
compare EVERY term to the VABS .K '6X6 Mass Matrix' (the validated ground truth). Resolves the
shell-vs-solid M56 sign question: get_mass_solid uses M[4,5]=+i23; VABS/get_mass_shell use the
negative product of inertia. area-integral over tri/quad elements; density per element from its set.'''
import os
import re

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SY = os.path.join(HERE, '2d_yaml', 'iea_r0020_solid.yaml')
KF = os.path.join(HERE, 'sg', 'iea_r0020.sg.K')
WIN = os.path.join(HERE, 'IEA-22-280-RWT.yaml')


def rows(raw, cast):
    out = []
    for r in raw:
        toks = r.split() if isinstance(r, str) else (r[0].split() if (len(r) == 1 and isinstance(r[0], str)) else r)
        out.append([cast(t) for t in toks])
    return out


d = yaml.safe_load(open(SY))
nd = np.array(rows(d['nodes'], float))
el = rows(d['elements'], int)                       # tri (3) or quad (4), 1-indexed
mats = d.get('materials', [])
# density per material name
dens = {}
for m in mats:
    nm = m.get('name')
    rho = m.get('density', m.get('rho'))
    if isinstance(rho, (list, tuple)):
        rho = rho[0]
    if rho is not None:
        dens[nm] = float(rho)
# windIO fallback for any missing density
wd = yaml.safe_load(open(WIN))
for m in wd['materials']:
    dens.setdefault(m['name'], float(m.get('rho', 0.0)))

# element -> material name via sections/sets
sec = d.get('sections', [])
setname_to_mat = {}
for s in sec:
    setname_to_mat[s['elementSet']] = s.get('material', s.get('name'))
el_mat = [None] * len(el)
for grp in d['sets']['element']:
    mat = setname_to_mat.get(grp['name'], grp['name'])
    for lab in grp['labels']:
        el_mat[int(lab) - 1] = mat

# integrate mu, first & second moments over the mesh
mu = xm2n = xm3n = i22 = i33 = i23 = 0.0
miss = set()
for k, e in enumerate(el):
    idx = [i - 1 for i in e]
    P = nd[idx][:, :2]
    if len(idx) == 3:
        A = 0.5 * abs((P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1]))
    else:
        x, y = P[:, 0], P[:, 1]
        A = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    c = P.mean(0)
    rho = dens.get(el_mat[k])
    if rho is None:
        miss.add(el_mat[k]); rho = 0.0
    m = rho * A
    mu += m; xm2n += c[0] * m; xm3n += c[1] * m
    i22 += c[1] ** 2 * m; i33 += c[0] ** 2 * m; i23 += c[0] * c[1] * m
xm2, xm3 = xm2n / mu, xm3n / mu
Msolid = np.array([[mu, 0, 0, 0, mu * xm3, -mu * xm2],
                   [0, mu, 0, -mu * xm3, 0, 0],
                   [0, 0, mu, mu * xm2, 0, 0],
                   [0, -mu * xm3, mu * xm2, i22 + i33, 0, 0],
                   [mu * xm3, 0, 0, 0, i22, i23],
                   [-mu * xm2, 0, 0, 0, i23, i33]])

# parse VABS .K 6x6 mass matrix
txt = open(KF).read().splitlines()
i0 = next(i for i, l in enumerate(txt) if 'The 6X6 Mass Matrix' in l)
MK = np.array([[float(x) for x in txt[i0 + 3 + j].split()] for j in range(6)])

print('r0020 solid mass: %d nodes, %d elems; missing-density mats: %s' % (len(nd), len(el), miss or 'none'))
print('%-8s %15s %15s %9s' % ('term', 'solid(get_mass)', 'VABS .K', '%err'))
for (i, j, nm) in [(0, 0, 'M11'), (0, 4, 'M15'), (0, 5, 'M16'), (3, 3, 'M44'),
                   (4, 4, 'M55'), (5, 5, 'M66'), (4, 5, 'M56')]:
    a, b = Msolid[i, j], MK[i, j]
    e = 100 * (a - b) / b if abs(b) > 1e-9 else 0.0
    print('%-8s %15.6e %15.6e %+8.2f' % (nm, a, b, e))
print('\nmass center solid = (%.4f, %.5f)  |  VABS = (%.4f, %.5f)'
      % (xm2, xm3, -MK[0, 5] / MK[0, 0], MK[0, 4] / MK[0, 0]))
print('i23 (product of inertia) = %.4e ; VABS M[4,5] = %.4e  -> solid uses +i23, sign %s VABS'
      % (i23, MK[4, 5], 'MATCHES' if np.sign(i23) == np.sign(MK[4, 5]) else 'OPPOSITE to'))
