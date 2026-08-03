'''confirm build_rm_bundle now defaults to mid-ref (center, frac=0.5).'''
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, '..', '..', 'TW-paper', 'xsec_paper')))
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm
SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')
Bd = dehom_rm.build_rm_bundle(SHELL)                 # NO ref -> should be center (mid-ref)
Bc = dehom_rm.build_rm_bundle(SHELL, ref='center')
print('default (no ref)  frac =', Bd['frac'])
print('explicit center   frac =', Bc['frac'])
print('default == explicit center 6x6 :', np.allclose(Bd['Timo'], Bc['Timo']))
print('Timo diag (default) =', np.array2string(np.diag(Bd['Timo']), precision=3))
