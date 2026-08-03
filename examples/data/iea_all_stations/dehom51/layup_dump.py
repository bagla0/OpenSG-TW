'''dump the through-thickness ply stack (z from -t/2 outer to +t/2 inner) for the cap + sandwich
layups, and mark where each thickness-path point's RM depth z lands -- to see if the carbon ply is
mis-positioned (ordering) or the path simply drifts off the single wall.'''
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm
SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')
B = dehom_rm.build_rm_bundle(SHELL, ref='center')
ldb = B['layup_db']
for ln in ('layup_2', 'layup_5'):
    i = ldb[ln]
    th = np.asarray(i['thick']); mats = i['mat_names']; ang = i.get('angles')
    t = th.sum()
    zlo = -t / 2.0
    print('\n%s  total=%.4f m  (n=%d)  ply stack from z=-t/2 (outer) -> +t/2 (inner):' % (ln, t, len(th)))
    z = zlo
    for k in range(len(th)):
        print('   ply %d  %-20s  t=%7.4f m   z:[%+.4f, %+.4f]  (%+.1f..%+.1f mm)  ang=%s'
              % (k, mats[k], th[k], z, z + th[k], z * 1e3, (z + th[k]) * 1e3,
                 (ang[k] if ang is not None else '-')))
        z += th[k]
