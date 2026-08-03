'''check the 3 outlier stations: shell vs solid diagonal + the 2d-yaml x2 range (origin) + node count.'''
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def d6(p):
    M = np.loadtxt(p)
    return [M[k, k] for k in range(6)]


def x2rng(p):
    d = yaml.safe_load(open(p))
    xs = []
    for r in d['nodes']:
        t = (r if isinstance(r, str) else r[0]).split()
        xs.append(float(t[0]))
    return min(xs), max(xs), len(xs)


for tag in ('s04', 's12', 's28'):
    sh = d6(os.path.join(HERE, 'homo_rm', 'OpenSG_RM_iea_%s.txt' % tag))
    so = d6(os.path.join(HERE, 'homo_jax', 'OpenSG_JAX_iea_%s.txt' % tag))
    print('\n=== %s ===' % tag)
    print('%-5s %12s %12s %8s' % ('', 'shell', 'solid', '%err'))
    for k in range(6):
        print('%-5s %12.4e %12.4e %+8.1f' % (LBL[k], sh[k], so[k], 100 * (sh[k] - so[k]) / so[k]))
    smin, smax, sn = x2rng(os.path.join(HERE, '1d_yaml', 'iea_%s_shell.yaml' % tag))
    omin, omax, on = x2rng(os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % tag))
    print('shell x2 [%.3f, %.3f] n=%d   solid x2 [%.3f, %.3f] n=%d' % (smin, smax, sn, omin, omax, on))
    mk = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml.refaxis' % tag)
    print('solid refaxis marker: %s' % (open(mk).read().strip() if os.path.exists(mk) else 'NONE'))
