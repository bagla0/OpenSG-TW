'''s28 solid (JAX) GA2/GA3 doubling. Re-homogenize s28 solid fresh (reproducibility) and
compare the s28 solid mesh (2d yaml) node/elem count + bbox vs neighbours s27,s29.'''
import os
import sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def mesh_info(tag):
    p = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % tag)
    d = yaml.safe_load(open(p))
    nodes = []
    for r in d['nodes']:
        t = (r if isinstance(r, str) else r[0]).split()
        nodes.append([float(t[0]), float(t[1])])
    nodes = np.array(nodes)
    ne = 0
    for key in ('elements', 'element_connectivity', 'connectivity'):
        if key in d:
            ne = len(d[key])
            break
    x2 = nodes[:, 0]
    x3 = nodes[:, 1]
    return dict(nn=len(nodes), ne=ne,
                x2=(x2.min(), x2.max()), x3=(x3.min(), x3.max()),
                keys=list(d.keys()))


print('=== s28 solid mesh vs neighbours ===')
for tag in ('s27', 's28', 's29'):
    mi = mesh_info(tag)
    print('%s nodes=%d elems=%d x2[%.3f,%.3f] x3[%.3f,%.3f]' % (
        tag, mi['nn'], mi['ne'], mi['x2'][0], mi['x2'][1], mi['x3'][0], mi['x3'][1]))
    print('   keys=%s' % mi['keys'])

# reproducibility: re-run s27/s28/s29 solid homogenization fresh
print('\n=== re-homogenize s27/s28/s29 solid (fresh) ===')
REPO = os.path.expanduser('~/OpenSG-TW-claude')
for q in (REPO, os.path.join(REPO, 'opensg_jax')):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import jax
jax.config.update('jax_enable_x64', True)
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
for tag in ('s27', 's28', 's29'):
    p = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % tag)
    K = np.asarray(compute_timo_from_yaml(p, verbose=False))
    print('%s fresh: EA=%.4e GA2=%.4e GA3=%.4e GJ=%.4e EI2=%.4e EI3=%.4e' % (
        tag, K[0, 0], K[1, 1], K[2, 2], K[3, 3], K[4, 4], K[5, 5]))
