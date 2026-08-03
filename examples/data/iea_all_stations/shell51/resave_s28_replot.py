'''Recompute the repaired s28 solid, overwrite homo_jax/OpenSG_JAX_iea_s28.txt, re-emit its .out,
then re-run the %-error plot.'''
import os
import sys
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
for q in (REPO, os.path.join(REPO, 'opensg_jax')):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import jax
jax.config.update('jax_enable_x64', True)
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml

p = os.path.join(HERE, '2d_yaml', 'iea_s28_solid.yaml')
K = np.asarray(compute_timo_from_yaml(p, verbose=False))
out = os.path.join(HERE, 'homo_jax', 'OpenSG_JAX_iea_s28.txt')
np.savetxt(out, K)
print('saved repaired s28 solid -> %s' % out)
print('  GA2=%.4e GA3=%.4e' % (K[1, 1], K[2, 2]))

PY = sys.executable
subprocess.run([PY, os.path.join(HERE, 'emit_full_out51.py'), '--source', 'jax'], check=False)
subprocess.run([PY, os.path.join(HERE, 'plot_pcterr.py')], check=False)
print('DONE')
