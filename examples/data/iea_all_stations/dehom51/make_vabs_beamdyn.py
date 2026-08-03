'''make_vabs_beamdyn.py -- BeamDyn from the VABS .K Timoshenko 6x6 (at (0,0)) + VABS mass 6x6,
for all 51 stations. .K in dehom51/out/VABS_iea51/. Missing stations linearly interpolated.
Writes props/primary/driver into VABS_iea51/, runs beamdyn_driver, then gen_glb reads the .out.'''
import os, sys, glob, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
BD = os.path.join(ROOT, 'beamdyn_iea')
VK = os.path.join(HERE, 'out', 'VABS_iea51')
sys.path.insert(0, BD)
from beamdyn_trans import transformMatrixToBeamDyn, write_beamdyn_prop

NSTA = 51
ETAS = np.arange(NSTA) / 50.0
BDDRV = '/home/roger/a/bagla0/miniconda3/bin/beamdyn_driver'
ENV = dict(os.environ); ENV['LD_LIBRARY_PATH'] = '/home/roger/a/bagla0/miniconda3/lib'


def block6(L, key):
    for i, l in enumerate(L):
        if key.lower() in l.lower():
            rows = []; j = i + 1
            while len(rows) < 6 and j < len(L):
                try:
                    v = [float(x) for x in L[j].split()]
                    if len(v) >= 6:
                        rows.append(v[:6])
                except ValueError:
                    pass
                j += 1
            if len(rows) == 6:
                return np.array(rows)
    return None


K = np.full((NSTA, 6, 6), np.nan); M = np.full((NSTA, 6, 6), np.nan)
present = []
for f in sorted(glob.glob(os.path.join(VK, '*.sg.K'))):
    tag = os.path.basename(f).replace('.sg.K', '')                    # iea_sNN
    i = int(tag.replace('iea_s', ''))
    L = open(f).read().splitlines()
    K[i] = block6(L, 'Timoshenko Stiffness Matrix')
    M[i] = block6(L, 'The 6X6 Mass Matrix')                           # first = at origin (0,0)
    present.append(i)
present = sorted(present)
missing = [i for i in range(NSTA) if i not in present]
print('VABS .K present: %d stations %s' % (len(present), present))
print('missing (interpolated): %s' % missing)
for i in missing:
    lo = max([j for j in present if j < i], default=None)
    hi = min([j for j in present if j > i], default=None)
    if lo is not None and hi is not None:
        w = (i - lo) / (hi - lo); K[i] = (1 - w) * K[lo] + w * K[hi]; M[i] = (1 - w) * M[lo] + w * M[hi]
    else:
        j = lo if lo is not None else hi; K[i] = K[j]; M[i] = M[j]

np.savetxt(os.path.join(VK, 'vabs_K6x6_51.dat'), K.reshape(NSTA, 36), fmt='%.8e',
           header='VABS .K Timoshenko 6x6 (VABS order, (0,0) origin), 51 stations')

Kb, Mb = transformMatrixToBeamDyn(K.copy(), M.copy())
pf = write_beamdyn_prop(VK, 'iea51vabs', ETAS, Kb, Mb, [1e-3] * 6)
os.replace(os.path.join(VK, pf), os.path.join(VK, 'iea51vabs_bd_props.inp'))
prim = open(os.path.join(BD, 'iea51_solid_bd_primary.inp')).read().replace(
    'iea51_solid_bd_props.inp', 'iea51vabs_bd_props.inp')
open(os.path.join(VK, 'iea51vabs_bd_primary.inp'), 'w').write(prim)
drv = open(os.path.join(BD, 'iea51_solid_bd_driver.inp')).read().replace(
    'iea51_solid_bd_primary.inp', 'iea51vabs_bd_primary.inp')
open(os.path.join(VK, 'iea51vabs_bd_driver.inp'), 'w').write(drv)
print('wrote iea51vabs props/primary/driver in VABS_iea51/')

r = subprocess.run([BDDRV, 'iea51vabs_bd_driver.inp'], cwd=VK, env=ENV, capture_output=True, text=True, timeout=600)
ok = os.path.exists(os.path.join(VK, 'iea51vabs_bd_driver.out'))
print('ran iea51vabs BeamDyn : %s' % ('OK' if ok else 'FAIL\n' + r.stdout[-800:] + r.stderr[-800:]))
if ok:
    L = [l for l in open(os.path.join(VK, 'iea51vabs_bd_driver.out')).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith('Time'):
            h = l.split(); row = np.array([r.split() for r in L[i + 2:]], float)[-1]
            N = sum(1 for x in h if x.endswith('_TDxr') and x.startswith('N'))
            print('  output nodes=%d  TipTDxr=%.4f m' % (N, row[h.index('TipTDxr')]))
            break
