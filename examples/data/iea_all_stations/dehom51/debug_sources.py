'''trace EVERY homo source for r0.2 (s10) to find where the RM-vs-JAX convention diverges:
  homo_rm/.txt   (raw RM homogenizer)          vs  homo_jax/.txt   (raw JAX solid)   <- the previous <5%
  out/OpenSG_RM_Shell/.out (BeamDyn input)     vs  out/OpenSG_JAX_Solid/.out
  build_rm_bundle(center)  (the RM dehom ring)
Print each diagonal + the pairwise %errors so we see if the gap is in the homo or the .out transform.'''
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm
S = 'iea_s10'
lbl = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def txt6(p):
    return np.loadtxt(p) if os.path.exists(p) else None


def out6(p):
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
            return np.array(rows)


src = {}
src['homo_rm.txt'] = txt6(os.path.join(ROOT, 'shell51/homo_rm/OpenSG_RM_%s.txt' % S))
src['homo_jax.txt'] = txt6(os.path.join(ROOT, 'shell51/homo_jax/OpenSG_JAX_%s.txt' % S))
src['RM_Shell.out'] = out6(os.path.join(ROOT, 'shell51/out/OpenSG_RM_Shell/%s_OpenSG_RM_Shell.out' % S))
src['JAX_Solid.out'] = out6(os.path.join(ROOT, 'shell51/out/OpenSG_JAX_Solid/%s_OpenSG_JAX_Solid.out' % S))
src['build_rm(center)'] = np.asarray(dehom_rm.build_rm_bundle(
    os.path.join(ROOT, 'shell51/1d_yaml/%s_shell.yaml' % S), ref='center')['Timo'])

print('%-18s %s' % ('source', '  '.join('%11s' % t for t in lbl)))
for nm, K in src.items():
    if K is None:
        print('%-18s MISSING' % nm); continue
    print('%-18s %s' % (nm, '  '.join('%11.4e' % d for d in np.diag(K))))


def pe(a, b):
    da, db = np.diag(a), np.diag(b)
    return '  '.join('%+7.1f' % (100 * (x - y) / y) for x, y in zip(da, db))


print('\n%-34s %s' % ('%err (per diag term)', '  '.join('%7s' % t for t in lbl)))
pairs = [('homo_rm.txt', 'homo_jax.txt', 'RAW homo: RM vs JAX   (<5% expected)'),
         ('RM_Shell.out', 'JAX_Solid.out', 'BeamDyn .out: RM vs JAX'),
         ('homo_rm.txt', 'RM_Shell.out', 'RM: raw homo vs its .out  (transform?)'),
         ('homo_jax.txt', 'JAX_Solid.out', 'JAX: raw homo vs its .out (transform?)'),
         ('build_rm(center)', 'homo_rm.txt', 'RM: build_rm_bundle vs raw homo_rm')]
for a, b, desc in pairs:
    if src.get(a) is not None and src.get(b) is not None:
        print('%-34s %s' % (desc, pe(src[a], src[b])))
