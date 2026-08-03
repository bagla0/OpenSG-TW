'''Homogenize the two robust-pipeline outputs (s02 via conditioned-PreVABS, s50 via fallback)
and confirm the Timo 6x6 diagonal sits on the spanwise trend of the neighbouring stations.'''
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
for q in (HERE, REPO, os.path.join(REPO, 'opensg_jax')):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import repair_mesh as RM
import jax
jax.config.update('jax_enable_x64', True)
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml

LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
ROB = os.path.join(HERE, 'shell51', 'robust_yaml')


def d6(p):
    M = np.loadtxt(p)
    return [M[k, k] for k in range(6)]


print('=== degeneracy scan of robust outputs ===')
for t in ('s02', 's50'):
    p = os.path.join(ROB, 'iea_%s_solid.yaml' % t)
    s = RM.diagnose(p)
    print('  %s: nn=%d ne=%d coincident=%d zero=%d slivers=%d' %
          (t, s['nn'], s['ne'], s['coincident'], s['zero_measure'], s['slivers']))

print('\n=== homogenize robust outputs (JAX solid) ===')
for t in ('s02', 's50'):
    p = os.path.join(ROB, 'iea_%s_solid.yaml' % t)
    K = np.asarray(compute_timo_from_yaml(p, verbose=False))
    print('  [%s] EA=%.4e GA2=%.4e GA3=%.4e GJ=%.4e EI2=%.4e EI3=%.4e' %
          (t, K[0, 0], K[1, 1], K[2, 2], K[3, 3], K[4, 4], K[5, 5]))

print('\n=== neighbours for trend (origin-independent EA/GA2/GA3/EI2) ===')
for t in ('s01', 's03', 's48', 's49'):
    p = os.path.join(HERE, 'shell51', 'homo_jax', 'OpenSG_JAX_iea_%s.txt' % t)
    if os.path.exists(p):
        k = d6(p)
        print('  [%s] EA=%.4e GA2=%.4e GA3=%.4e EI2=%.4e' % (t, k[0], k[1], k[2], k[4]))
print('DONE')
