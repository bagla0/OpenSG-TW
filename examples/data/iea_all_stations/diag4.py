'''diag4.py -- isolate the OML-vs-mid degradation. For r0247 (mid airfoil) and r0000 (thick root),
homogenize the mid-surface mesh (1d_yaml) and the OML mesh (1d_yaml_oml) each with center_ref
True/False, and compare EA/GJ (origin-independent) to the 2-D solid. Tells us whether the contour
(fraction) or the ABD reference (center_ref) drives the difference.'''
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XS = os.path.expanduser('~/OpenSG-TW-claude/examples/TW-paper/xsec_paper')
REPO = os.path.abspath(os.path.join(XS, '..', '..', '..'))
for q in (XS, REPO, os.path.join(REPO, 'mitc_rm_segment')):
    sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from xsec_5v6_master import load_ring, ring_6dof


def read_stiff(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith('Stiffness'))
    return np.array([[float(x) for x in lines[i + 1 + j].split()] for j in range(6)])


for tag in ('r0247', 'r0000'):
    mid = os.path.join(HERE, '1d_yaml', 'iea_%s_shell.yaml' % tag)
    oml = os.path.join(HERE, '1d_yaml_oml', 'iea_%s_shell.yaml' % tag)
    sol = read_stiff(os.path.join(HERE, 'out', 'OpenSG_FEniCSx_Solid',
                                  'iea_%s_OpenSG_FEniCSx_Solid.out' % tag))
    EA_s, GJ_s = sol[0, 0], sol[3, 3]
    print('\n=== %s : EA_solid=%.4e  GJ_solid=%.4e ===' % (tag, EA_s, GJ_s))
    print('%-28s %12s %8s %12s %8s' % ('config', 'EA', 'EA%', 'GJ', 'GJ%'))
    for name, path, cr in [('mid contour + center_ref=T', mid, True),
                           ('mid contour + center_ref=F', mid, False),
                           ('OML contour + center_ref=T', oml, True),
                           ('OML contour + center_ref=F', oml, False)]:
        C = np.asarray(ring_6dof(load_ring(path, center_ref=cr)))
        EA, GJ = C[0, 0], C[3, 3]
        print('%-28s %12.4e %+7.1f %12.4e %+7.1f'
              % (name, EA, 100 * (EA - EA_s) / EA_s, GJ, 100 * (GJ - GJ_s) / GJ_s))
