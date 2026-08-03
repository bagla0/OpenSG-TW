'''validate_oml.py -- homogenize the OML + reference-axis 1-D shell rings (center_ref=False) and
compare the Timoshenko 6x6 diagonal against (i) the old mid-surface/LE RM .out and (ii) the 2-D
solid .out.  EA, GA2, GA3, GJ are ORIGIN-INDEPENDENT, so those validate the OML choice directly
(the solid is the OML reference).  Writes the new 6x6 to 1d_yaml_oml homo -> out_oml/.'''
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

OUT = os.path.join(HERE, 'out_oml'); os.makedirs(OUT, exist_ok=True)
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def read_stiff(path):
    '''parse the 6x6 Stiffness block from a *.out file'''
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith('Stiffness'))
    M = [[float(x) for x in lines[i + 1 + j].split()] for j in range(6)]
    return np.array(M)


def diag(M):
    return np.array([M[k, k] for k in range(6)])


files = sorted(glob.glob(os.path.join(HERE, '1d_yaml_oml', 'iea_r*_shell.yaml')))
print('%-7s %10s %10s %10s %10s %10s %10s   %8s' % ('tag', *LBL, 't[s]'))
newdiag = {}
for f in files:
    tag = os.path.basename(f).split('_')[1]
    t0 = time.time()
    C6 = np.asarray(ring_6dof(load_ring(f, center_ref=False)))
    np.savetxt(os.path.join(OUT, 'OpenSG_RM_OML_%s.txt' % tag), C6)
    dg = diag(C6); newdiag[tag] = dg
    print('%-7s %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e   %6.1f' % (tag, *dg, time.time() - t0))

# ---- compare EA/GJ (origin-independent) vs solid + old RM
print('\n=== EA & GJ vs 2-D solid (origin-independent) : %%err = 100*(shell-solid)/solid ===')
print('%-7s | %11s %11s %11s |  %6s %6s | %11s %11s |  %6s %6s'
      % ('tag', 'EA_oml', 'EA_mid', 'EA_solid', 'oml%', 'mid%', 'GJ_oml', 'GJ_mid', 'oml%', 'mid%'))
RMOLD = os.path.join(HERE, 'out', 'OpenSG_RM_Shell')
SOLID = os.path.join(HERE, 'out', 'OpenSG_FEniCSx_Solid')
for f in files:
    tag = os.path.basename(f).split('_')[1]
    dg = newdiag[tag]
    po = os.path.join(RMOLD, 'iea_%s_OpenSG_RM_Shell.out' % tag)
    ps = os.path.join(SOLID, 'iea_%s_OpenSG_FEniCSx_Solid.out' % tag)
    if not (os.path.exists(po) and os.path.exists(ps)):
        print('%-7s | (missing old RM or solid .out)' % tag); continue
    dm = diag(read_stiff(po)); dsd = diag(read_stiff(ps))
    ea_o, ea_m, ea_s = dg[0], dm[0], dsd[0]
    gj_o, gj_m, gj_s = dg[3], dm[3], dsd[3]
    print('%-7s | %11.4e %11.4e %11.4e | %+6.1f %+6.1f | %11.4e %11.4e | %+6.1f %+6.1f'
          % (tag, ea_o, ea_m, ea_s, 100 * (ea_o - ea_s) / ea_s, 100 * (ea_m - ea_s) / ea_s,
             gj_o, gj_m, 100 * (gj_o - gj_s) / gj_s, 100 * (gj_m - gj_s) / gj_s))
print('\nwrote 6x6 -> out_oml/')
