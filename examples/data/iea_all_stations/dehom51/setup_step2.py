'''setup_step2.py -- (1) fix the backtick-mangled VABS folder name -> VABS_iea51,
(2) show the VABS .SM / .U file format, (3) map solid-yaml Material_N -> real material name.'''
import os
import numpy as np
import yaml
BASE = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
OUT = os.path.join(BASE, 'dehom51', 'out')

# ---- (1) rename the mangled VABS folder ----
vdirs = [d for d in os.listdir(OUT) if d.startswith('VABS') and os.path.isdir(os.path.join(OUT, d))]
print('VABS dirs found:', [repr(d) for d in vdirs])
target = os.path.join(OUT, 'VABS_iea51')
for d in vdirs:
    full = os.path.join(OUT, d)
    kinds = {}
    for f in os.listdir(full):
        ext = f.split('.', 2)[-1]
        kinds[ext] = kinds.get(ext, 0) + 1
    print('  %-16s -> %d files  kinds=%s' % (repr(d), len(os.listdir(full)), kinds))
    if d != 'VABS_iea51' and not os.path.exists(target):
        os.rename(full, target)
        print('  RENAMED %r -> VABS_iea51' % d)
print('VABS_iea51 exists now:', os.path.isdir(target))

# ---- (2) VABS .SM / .U format ----
sm = os.path.join(target, 'iea_s10.sg.SM'); u = os.path.join(target, 'iea_s10.sg.U')
print('\n.SM head (iea_s10):')
for ln in open(sm).read().splitlines()[:3]:
    print('   ', ln[:120])
print('.U head (iea_s10):')
for ln in open(u).read().splitlines()[:3]:
    print('   ', ln[:120])
dsm = np.loadtxt(sm, skiprows=2)
du = np.loadtxt(u)
print('.SM shape', dsm.shape, ' cols: x y + stresses ; xy range x[%.3f,%.3f] y[%.3f,%.3f]'
      % (dsm[:, 0].min(), dsm[:, 0].max(), dsm[:, 1].min(), dsm[:, 1].max()))
print('.U  shape', du.shape, ' cols: id x y u1 u2 u3 ; xy range x[%.3f,%.3f] y[%.3f,%.3f]'
      % (du[:, 1].min(), du[:, 1].max(), du[:, 2].min(), du[:, 2].max()))


# ---- (3) material-name mapping ----
def frow(v):
    if isinstance(v, str):
        return [float(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [float(x) for x in v[0].split()]
    return [float(x) for x in v]


sol = yaml.safe_load(open(os.path.join(BASE, 'shell51/2d_yaml/iea_s10_solid.yaml')))
solmat = [(m['name'], frow(m['E'])[0], float(m['rho'])) for m in sol['materials']]
sh = yaml.safe_load(open(os.path.join(BASE, 'shell51/1d_yaml/iea_s10_shell.yaml')))
shmat = []
for m in sh['materials']:
    el = m.get('elastic', {})
    # elastic may hold 'E' list or a stiffness matrix; grab a representative E1 + density
    E1 = None
    if isinstance(el, dict):
        if 'E' in el:
            E1 = frow(el['E'])[0]
        elif 'youngs_modulus' in el:
            E1 = frow(el['youngs_modulus'])[0]
    rho = m.get('density', m.get('rho', None))
    shmat.append((m['name'], E1, float(rho) if rho is not None else None))
print('\nSHELL yaml materials (name, E1, rho):')
for nm, E1, rho in shmat:
    print('   %-22s E1=%s rho=%s' % (nm, E1, rho))
print('\nSOLID yaml Material_N -> best real name (match by rho, then E1):')
mapping = {}
for nm, E1s, rhos in solmat:
    best = None; bestd = 1e99
    for snm, E1h, rhoh in shmat:
        d = 0.0
        if rhoh is not None:
            d += abs(rhos - rhoh) / max(rhos, 1)
        if E1h is not None:
            d += abs(E1s - E1h) / max(E1s, 1)
        if d < bestd:
            bestd = d; best = snm
    mapping[nm] = best
    print('   %-14s (E1=%.3e rho=%.0f) -> %-22s' % (nm, E1s, rhos, best))
import json
open(os.path.join(BASE, 'dehom51', 'coords', 'material_names.json'), 'w').write(json.dumps(mapping, indent=2))
print('\nwrote coords/material_names.json')
