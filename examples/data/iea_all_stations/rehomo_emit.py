'''rehomo_emit.py -- re-homogenize the mid-surface + reference-axis 1-D shell rings (center_ref=True,
the validated config) and rewrite out/OpenSG_RM_Shell/iea_r*_OpenSG_RM_Shell.out (Stiffness +
Compliance, VABS order).  The 6x6 is now referenced to the windIO reference axis (x1), so it is
consistent with ff_beam_load.  EA/GA/GJ are origin-independent (unchanged, validated); EI2/EI3 are
now about the reference axis.'''
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XS = os.path.expanduser('~/OpenSG-TW-claude/examples/TW-paper/xsec_paper')
REPO = os.path.abspath(os.path.join(XS, '..', '..', '..'))
for q in (XS, REPO, os.path.join(REPO, 'mitc_rm_segment')):
    sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from xsec_5v6_master import load_ring, ring_6dof

OUTDIR = os.path.join(HERE, 'out', 'OpenSG_RM_Shell'); os.makedirs(OUTDIR, exist_ok=True)
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def mat(fh, name, M):
    fh.write('%s\n' % name)
    for r in M:
        fh.write('   ' + '   '.join('%.10e' % v for v in r) + '\n')


print('%-7s %10s %10s %10s %10s %10s %10s  %5s' % ('tag', *LBL, 't[s]'))
for f in sorted(glob.glob(os.path.join(HERE, '1d_yaml', 'iea_r*_shell.yaml'))):
    tag = os.path.basename(f).split('_')[1]
    t0 = time.time()
    C6 = np.asarray(ring_6dof(load_ring(f, center_ref=True)))
    dt = time.time() - t0
    S = 0.5 * (C6 + C6.T)
    comp = np.linalg.inv(S)
    with open(os.path.join(OUTDIR, 'iea_%s_OpenSG_RM_Shell.out' % tag), 'w') as fh:
        fh.write('# Timoshenko 6x6 -- RM SHELL cross-section (OpenSG-RM, mid-surface, reference-axis origin)\n')
        fh.write('# convention (VABS/OpenSG order): 1=extension, 2-3=transverse shear, 4=torsion, 5-6=bending\n')
        fh.write('# origin = windIO reference axis (x1); Time-taken: %.2f s\n\n' % dt)
        mat(fh, 'Stiffness :', S)
        fh.write('\n')
        mat(fh, 'Compliance:', comp)
    print('%-7s %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e  %5.1f'
          % (tag, *[S[k, k] for k in range(6)], dt))
print('\nwrote out/OpenSG_RM_Shell/*.out (mid-surface, reference-axis origin)')
