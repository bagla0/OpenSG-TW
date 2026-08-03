'''finalize_refaxis.py -- move ALL cross-section geometry to the windIO reference axis (x1):
shift every 1-D shell (1d_yaml) AND 2-D solid (2d_yaml) mesh by -section_offset_y(eta) [rigid
translation, in place], then re-homogenize both (RM shell + JAX solid) and rewrite the .out files.
The reference axis is now the single origin for the load, the shell 6x6 and the solid 6x6.
EA/GA/EI2 are unchanged; GJ/EI3 move to x1 (validated: shell(x1) ~ solid(x1) to ~1%).
Run ONCE (it shifts in place). Meshes were at the LE (git-committed / freshly generated) beforehand.
    ~/miniconda3/envs/opensg_2_0/bin/python finalize_refaxis.py
'''
import glob
import os
import sys
import time

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

ETA = {'r0000': 0.0, 'r0020': 0.02, 'r0049': 0.04868313468735217, 'r0066': 0.06649308419703373,
       'r0083': 0.08345591801468194, 'r0102': 0.10220904193570582, 'r0110': 0.11036272014164386,
       'r0136': 0.1364246061019374, 'r0156': 0.15564440587515185, 'r0197': 0.19665336575444797,
       'r0247': 0.24696148735364706, 'r0399': 0.3992636115637571, 'r0534': 0.5335887750152993,
       'r0739': 0.738938689884722, 'r0980': 0.9799991709122947, 'r1000': 1.0}
_d = yaml.safe_load(open(os.path.join(HERE, 'IEA-22-280-RWT.yaml')))
_so = _d['components']['blade']['outer_shape']['section_offset_y']
SO_G, SO_V = np.array(_so['grid']), np.array(_so['values'])
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def section_offset_y(e):
    return float(np.interp(e, SO_G, SO_V))


def shift_nodes_inplace(f, dx, fmt='%.8f'):
    lines = open(f).read().splitlines()
    out, inn = [], False
    for ln in lines:
        s = ln.strip()
        if s == 'nodes:':
            inn = True; out.append(ln); continue
        if inn and not s.startswith('-'):
            inn = False
        if inn and s.startswith('-'):
            t = s[1:].strip().strip('[]').split()
            out.append(('- [' + fmt + ' ' + fmt + ' ' + fmt + ']') % (float(t[0]) - dx, float(t[1]), float(t[2])))
        else:
            out.append(ln)
    open(f, 'w').write('\n'.join(out) + '\n')


def write_out(path, header_kind, S, dt):
    comp = np.linalg.inv(S)
    with open(path, 'w') as fh:
        fh.write('# Timoshenko 6x6 -- %s (reference-axis origin x1)\n' % header_kind)
        fh.write('# convention (VABS/OpenSG order): 1=extension, 2-3=transverse shear, 4=torsion, 5-6=bending\n')
        fh.write('# origin = windIO reference axis (x1); Time-taken: %.2f s\n\n' % dt)
        fh.write('Stiffness :\n')
        for r in S:
            fh.write('   ' + '   '.join('%.10e' % v for v in r) + '\n')
        fh.write('\nCompliance:\n')
        for r in comp:
            fh.write('   ' + '   '.join('%.10e' % v for v in r) + '\n')


RM = os.path.join(HERE, 'out', 'OpenSG_RM_Shell')
JX = os.path.join(HERE, 'out', 'OpenSG_JAX_Solid')
for p in (RM, JX):
    os.makedirs(p, exist_ok=True)

print('%-7s %8s | %-9s %-9s %-9s | %-9s %-9s %-9s' % ('tag', 'offy', 'EA_sh', 'GJ_sh', 'EI3_sh', 'EA_so', 'GJ_so', 'EI3_so'))
for tag in sorted(ETA, key=lambda t: ETA[t]):
    offy = section_offset_y(ETA[tag])
    sh = os.path.join(HERE, '1d_yaml', 'iea_%s_shell.yaml' % tag)
    so = os.path.join(HERE, '2d_yaml', 'iea_%s_solid.yaml' % tag)
    shift_nodes_inplace(sh, offy, '%.8f')
    t0 = time.time(); Ksh = 0.5 * (lambda C: C + C.T)(np.asarray(ring_6dof(load_ring(sh, center_ref=True)))); tsh = time.time() - t0
    write_out(os.path.join(RM, 'iea_%s_OpenSG_RM_Shell.out' % tag), 'RM SHELL cross-section (OpenSG-RM, mid-surface)', Ksh, tsh)
    line = '%-7s %8.3f | %.3e %.3e %.3e' % (tag, offy, Ksh[0, 0], Ksh[3, 3], Ksh[5, 5])
    if os.path.exists(so):
        shift_nodes_inplace(so, offy, '%.6f')
        t0 = time.time(); Kso = 0.5 * (lambda C: C + C.T)(np.asarray(compute_timo_from_yaml(so, verbose=False))); tso = time.time() - t0
        write_out(os.path.join(JX, 'iea_%s_OpenSG_JAX_Solid.out' % tag), '2-D SOLID cross-section (OpenSG-JAX)', Kso, tso)
        line += ' | %.3e %.3e %.3e' % (Kso[0, 0], Kso[3, 3], Kso[5, 5])
    else:
        line += ' | (no 2d solid)'
    print(line, flush=True)
print('\ndone: 1d_yaml + 2d_yaml shifted to x1; out/OpenSG_RM_Shell + out/OpenSG_JAX_Solid rewritten')
