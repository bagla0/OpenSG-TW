'''Ground-truth probe: which stations have shell+solid yaml+homogenized txt; do s02/s50 exist;
node counts; and the current s28 saved solid GA (post-repair check).'''
import os
import glob
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def toks(r):
    if isinstance(r, str):
        return r.split()
    if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
        return r[0].split()
    return [str(x) for x in r]


def nn(p):
    try:
        d = yaml.safe_load(open(p))
        return len(d['nodes'])
    except Exception:
        return -1


solids = sorted(glob.glob(os.path.join(HERE, '2d_yaml', '*_solid.yaml')))
shells = sorted(glob.glob(os.path.join(HERE, '1d_yaml', '*_shell.yaml')))
tags = sorted(set(os.path.basename(p).replace('iea_', '').replace('_solid.yaml', '').replace('_shell.yaml', '')
                  for p in solids + shells))
print('total tags: %d' % len(tags))
print('%-6s %6s %6s %8s %8s' % ('tag', 'shell', 'solid', 'RMtxt', 'JAXtxt'))
for t in tags:
    sp = os.path.join(HERE, '1d_yaml', 'iea_%s_shell.yaml' % t)
    op = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % t)
    rm = os.path.join(HERE, 'homo_rm', 'OpenSG_RM_iea_%s.txt' % t)
    jx = os.path.join(HERE, 'homo_jax', 'OpenSG_JAX_iea_%s.txt' % t)
    print('%-6s %6s %6s %8s %8s' % (
        t,
        nn(sp) if os.path.exists(sp) else 'MISS',
        nn(op) if os.path.exists(op) else 'MISS',
        'ok' if os.path.exists(rm) else '--',
        'ok' if os.path.exists(jx) else '--'))

# focus on s02/s50/s28
print('\n--- focus s02 / s50 / s28 ---')
for t in ('s02', 's50', 's28'):
    op = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % t)
    orig = op + '.orig'
    jx = os.path.join(HERE, 'homo_jax', 'OpenSG_JAX_iea_%s.txt' % t)
    line = 's%s: solid=%s' % (t, 'yes n=%d' % nn(op) if os.path.exists(op) else 'MISSING')
    if os.path.exists(orig):
        line += ' (has .orig backup n=%d)' % nn(orig)
    if os.path.exists(jx):
        K = np.loadtxt(jx)
        line += '  JAX GA2=%.3e GA3=%.3e' % (K[1, 1], K[2, 2])
    print(line)
