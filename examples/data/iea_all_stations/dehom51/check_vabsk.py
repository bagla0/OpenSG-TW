'''compare the solid homogenizers vs VABS .K (Timoshenko 6x6) for whatever stations have a .K.
report per-term %err so we see if the gap is origin-dependent (GJ/EI) or transverse shear (GA3).'''
import os, sys, glob
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']

print('=== VABS .K files present ===')
ks = sorted(glob.glob(os.path.join(ROOT, 'dehom_iea/sg_v201/*.sg.K')) +
            glob.glob(os.path.join(ROOT, 'sg/*.sg.K')) +
            glob.glob(os.path.join(ROOT, '../2d_yaml/*.sg.K')) +
            glob.glob(os.path.join(ROOT, '../2d_yaml/IEA_VABS/*.sg.K')))
for k in ks[:20]:
    print('   ', os.path.relpath(k, ROOT))
print('   total:', len(ks))

print('\n=== FEniCS-OpenSG solid homogenizer available? ===')
for f in ['homo_fenics_solid.py', 'homo_fenics.py']:
    p = os.path.join(ROOT, f)
    print('   %-24s %s' % (f, 'YES' if os.path.exists(p) else 'no'), p if os.path.exists(p) else '')
# fenics env
import subprocess
for env in ['opensg_env_v8', 'opensg_env', 'jax-fem-env']:
    dol = '/home/roger/a/bagla0/miniconda3/envs/%s/lib' % env
    has = any('dolfinx' in d for d in (os.listdir('/home/roger/a/bagla0/miniconda3/envs/%s/lib/python3.11/site-packages' % env)
              if os.path.isdir('/home/roger/a/bagla0/miniconda3/envs/%s/lib/python3.11/site-packages' % env) else []))


def vabs_timo(p):
    L = open(p).read().splitlines()
    for i, l in enumerate(L):
        if 'timoshenko stiffness' in l.lower():
            rows = []
            for l2 in L[i + 1:]:
                try:
                    v = [float(x) for x in l2.split()]
                    if len(v) >= 6:
                        rows.append(v[:6])
                except ValueError:
                    pass
                if len(rows) == 6:
                    break
            if len(rows) == 6:
                return np.array(rows)
    return None


def outdiag(sub, i):
    p = os.path.join(ROOT, 'shell51/out', sub, 'iea_s%02d_%s.out' % (i, sub))
    if not os.path.exists(p):
        return None
    L = open(p).read().splitlines()
    for j, l in enumerate(L):
        if l.strip().startswith('Stiffness'):
            rows = []; k = j + 1
            while len(rows) < 6 and k < len(L):
                try:
                    fv = [float(x) for x in L[k].split()]
                    if len(fv) >= 6:
                        rows.append(fv[:6])
                except ValueError:
                    pass
                k += 1
            return np.diag(np.array(rows))


def txtdiag(sub, pref, i):
    p = os.path.join(ROOT, 'shell51', sub, '%s_iea_s%02d.txt' % (pref, i))
    return np.diag(np.loadtxt(p)) if os.path.exists(p) else None


print('\n=== solid homo vs VABS .K (Timoshenko diag) ===')
for k in ks:
    base = os.path.basename(k).replace('.sg.K', '')
    if not base.startswith('iea_s'):
        continue
    i = int(base.replace('iea_s', ''))
    V = vabs_timo(k)
    if V is None:
        print('  %s : no Timoshenko block in .K' % base); continue
    vd = np.diag(V)
    print('  %s VABS.K diag = %s' % (base, np.array2string(vd, precision=3)))
    for src, d in [('homo_jax.txt', txtdiag('homo_jax', 'OpenSG_JAX', i)),
                   ('JAX_Solid.out', outdiag('OpenSG_JAX_Solid', i)),
                   ('homo_rm.txt', txtdiag('homo_rm', 'OpenSG_RM', i))]:
        if d is not None:
            err = 100 * (d - vd) / vd
            print('      %-14s %%err vs VABS.K = [%s]' % (src, ' '.join('%+6.1f' % e for e in err)))
