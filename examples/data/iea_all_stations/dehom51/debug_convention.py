'''DEBUG the RM(1D shell) vs JAX(2D solid) convention for r0.2 (iea_s10):
frames/origin of the 1D shell yaml and 2D solid yaml, and the RM 6x6 at each reference
surface (oml/center/iml) vs the JAX-solid 6x6 -- to find which reference the center-ref
1D yaml actually needs.'''
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm
import yaml

SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')
SOLID = os.path.join(ROOT, 'shell51/2d_hybrid/iea_s10_solid.yaml')
JAXOUT = os.path.join(ROOT, 'shell51/out/OpenSG_JAX_Solid/iea_s10_OpenSG_JAX_Solid.out')
lbl = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def _row(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


def nodes(p):
    return np.array([_row(n)[:2] for n in yaml.safe_load(open(p))['nodes']])


sh, so = nodes(SHELL), nodes(SOLID)
print('1D shell : x[%.4f,%.4f] y[%.4f,%.4f]  node-centroid=(%.4f,%.4f)  n=%d' %
      (sh[:, 0].min(), sh[:, 0].max(), sh[:, 1].min(), sh[:, 1].max(), sh[:, 0].mean(), sh[:, 1].mean(), len(sh)))
print('2D solid : x[%.4f,%.4f] y[%.4f,%.4f]  node-centroid=(%.4f,%.4f)  n=%d' %
      (so[:, 0].min(), so[:, 0].max(), so[:, 1].min(), so[:, 1].max(), so[:, 0].mean(), so[:, 1].mean(), len(so)))
print('  origin offset (shell - solid) x=%.4f y=%.4f' % (0.0, 0.0))
print('  do both contain (0,0)?  shell x-range crosses 0: %s ; solid: %s'
      % (sh[:, 0].min() < 0 < sh[:, 0].max(), so[:, 0].min() < 0 < so[:, 0].max()))


def outdiag(p):
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


jx = outdiag(JAXOUT)
print('\nJAX-solid diag       = %s' % np.array2string(jx, precision=4))
print('%-16s %s   %s' % ('', 'diag(EA GA2 GA3 GJ EI2 EI3)', '%err vs JAX (per term)'))
for ref in ('oml', 'center', 'iml'):
    try:
        K = np.asarray(dehom_rm.build_rm_bundle(SHELL, ref=ref)['Timo'])
        d = np.diag(K); err = 100 * (d - jx) / jx
        print('RM ref=%-7s   %s   [%s] %%' % (ref, np.array2string(d, precision=4),
              ' '.join('%+5.1f' % e for e in err)))
    except Exception as e:
        print('RM ref=%-7s   FAILED: %r' % (ref, e))
