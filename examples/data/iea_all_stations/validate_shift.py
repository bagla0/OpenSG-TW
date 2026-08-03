'''validate_shift.py -- the CORRECT same-origin check for r0247. Compare RM-shell vs 2-D-solid
Timoshenko 6x6 at BOTH the LE origin AND the windIO reference-axis origin, by shifting the SAME
geometry for both models. Also cross-check re-homogenization vs the analytic parallel-axis transform
(translation-covariance). GJ/EI2/EI3 are origin-dependent; EA/GA2/GA3 are not.
    ~/miniconda3/envs/opensg_2_0/bin/python validate_shift.py
'''
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
XS = os.path.expanduser('~/OpenSG-TW-claude/examples/TW-paper/xsec_paper')
REPO = os.path.expanduser('~/OpenSG-TW-claude')
for q in (XS, REPO, os.path.join(REPO, 'opensg_jax'), os.path.join(REPO, 'mitc_rm_segment')):
    sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import jax
jax.config.update('jax_enable_x64', True)
from xsec_5v6_master import load_ring, ring_6dof
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml

LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']

# section_offset_y(0.247)
d = yaml.safe_load(open(os.path.join(HERE, 'IEA-22-280-RWT.yaml')))
so = d['components']['blade']['outer_shape']['section_offset_y']
OFFY = float(np.interp(0.24696148735364706, np.array(so['grid']), np.array(so['values'])))
print('reference-axis offset section_offset_y(r0247) = %.4f m' % OFFY)

shell_yaml = os.path.join(HERE, '1d_yaml', 'iea_r0247_shell.yaml')
solid_yaml = os.path.join(HERE, '2d_yaml', 'iea_r0247_solid.yaml')


def shift_solid_yaml(src, dst, dx):
    lines = open(src).read().splitlines()
    out, inn = [], False
    for ln in lines:
        s = ln.strip()
        if s == 'nodes:':
            inn = True; out.append(ln); continue
        if inn and not s.startswith('-'):
            inn = False
        if inn and s.startswith('-'):
            t = s[1:].strip().strip('[]').split()
            out.append('- [%.6f %.6f %.6f]' % (float(t[0]) - dx, float(t[1]), float(t[2])))
        else:
            out.append(ln)
    open(dst, 'w').write('\n'.join(out) + '\n')


# ---- shell RM: LE + reference-axis (in-memory node shift) ----
K_shell_LE = np.asarray(ring_6dof(load_ring(shell_yaml, center_ref=True)))
R = load_ring(shell_yaml, center_ref=True)
R['rx'][:, R['cross']] = R['rx'][:, R['cross']] - np.array([OFFY, 0.0])
K_shell_ref = np.asarray(ring_6dof(R))

# ---- 2-D solid: LE + reference-axis (shifted yaml) ----
K_solid_LE = np.asarray(compute_timo_from_yaml(solid_yaml, verbose=False))
sh = '/tmp/iea_r0247_solid_ref.yaml'
shift_solid_yaml(solid_yaml, sh, OFFY)
K_solid_ref = np.asarray(compute_timo_from_yaml(sh, verbose=False))

# ---- analytic parallel-axis (offset t2=OFFY, t3=0):  K' = A K A^T ----
A = np.eye(6); A[3, 2] = OFFY; A[5, 0] = -OFFY
Ksh_ref_PA = A @ K_shell_LE @ A.T
Kso_ref_PA = A @ K_solid_LE @ A.T


def dg(M):
    return np.array([M[k, k] for k in range(6)])


print('\n=== DIAGONAL 6x6 : shell-RM vs 2-D-solid, at each origin ===')
print('%-5s | %11s %11s %7s | %11s %11s %7s' % ('', 'shell_LE', 'solid_LE', 'd%', 'shell_ref', 'solid_ref', 'd%'))
for k in range(6):
    a, b, c, e = dg(K_shell_LE)[k], dg(K_solid_LE)[k], dg(K_shell_ref)[k], dg(K_solid_ref)[k]
    print('%-5s | %11.4e %11.4e %+6.1f | %11.4e %11.4e %+6.1f'
          % (LBL[k], a, b, 100 * (a - b) / b, c, e, 100 * (c - e) / e))

print('\n=== translation-covariance : re-homogenized(ref) vs parallel-axis(LE->ref) ===')
print('%-5s | %11s %11s %7s | %11s %11s %7s' % ('', 'shell_reh', 'shell_PA', 'd%', 'solid_reh', 'solid_PA', 'd%'))
for k in range(6):
    a, b, c, e = dg(K_shell_ref)[k], dg(Ksh_ref_PA)[k], dg(K_solid_ref)[k], dg(Kso_ref_PA)[k]
    print('%-5s | %11.4e %11.4e %+6.1f | %11.4e %11.4e %+6.1f'
          % (LBL[k], a, b, 100 * (a - b) / b if b else 0, c, e, 100 * (c - e) / e if e else 0))
