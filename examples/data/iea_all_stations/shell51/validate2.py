'''validate2.py -- shell(RM) vs solid(JAX) diagonal 6x6 on 2 stations (default s04, s12), all at the
x1 reference axis.  Confirms the shell51-generated shells agree with the shell51-generated solids to
~1-3% (as the committed 16 stations did) before the full 51 run.'''
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
XS = os.path.join(REPO, 'examples', 'TW-paper', 'xsec_paper')
for q in (XS, REPO, os.path.join(REPO, 'opensg_jax'), os.path.join(REPO, 'mitc_rm_segment')):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import jax
jax.config.update('jax_enable_x64', True)
from xsec_5v6_master import load_ring, ring_6dof
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml

LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
stations = sys.argv[1:] or ['iea_s04', 'iea_s12']

for st in stations:
    y1 = os.path.join(HERE, '1d_yaml', st + '_shell.yaml')
    y2 = os.path.join(HERE, '2d_yaml', st + '_solid.yaml')
    t0 = time.time()
    Csh = np.asarray(ring_6dof(load_ring(y1)))
    t1 = time.time()
    Cso = np.asarray(compute_timo_from_yaml(y2, verbose=False))
    t2 = time.time()
    dsh = np.diag(Csh); dso = np.diag(Cso)
    print('\n=== %s  (shell %.1fs, solid %.1fs) ===' % (st, t1 - t0, t2 - t1))
    print('%-5s %14s %14s %9s' % ('term', 'RM shell', 'JAX solid', '%err'))
    for j in range(6):
        e = 100.0 * (dsh[j] - dso[j]) / dso[j]
        print('%-5s %14.5e %14.5e %8.2f%%' % (LBL[j], dsh[j], dso[j], e))
