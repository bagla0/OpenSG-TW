'''Compare the converged BeamDyn static response of the SOLID vs SHELL cross-section models.'''
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['TipTDxr', 'TipTDyr', 'TipTDzr', 'TipRDzr', 'RootFxr', 'RootMxr', 'RootMyr', 'RootMzr']
LBL = {'TipTDxr': 'tip flap defl [m]', 'TipTDyr': 'tip edge defl [m]', 'TipTDzr': 'tip axial [m]',
       'TipRDzr': 'tip twist [rad]', 'RootFxr': 'root flap shear [N]', 'RootMxr': 'root torsion [N-m]',
       'RootMyr': 'root flap moment [N-m]', 'RootMzr': 'root edge moment [N-m]'}


def last_row(f):
    lines = [l for l in open(f).read().splitlines() if l.strip()]
    for i, l in enumerate(lines):
        if l.strip().startswith('Time'):
            hdr = l.split()
            data = np.array([r.split() for r in lines[i + 2:]], float)
            return {h: data[-1, j] for j, h in enumerate(hdr)}
    raise ValueError('no header')


sol = last_row(os.path.join(HERE, 'iea51_solid_bd_driver.out'))
she = last_row(os.path.join(HERE, 'iea51_shell_bd_driver.out'))
print('BeamDyn static response of the IEA-22 blade @ 1500 Pa flapwise (loads about x1)')
print('%-22s %16s %16s %9s' % ('quantity', 'SOLID (2-D)', 'SHELL (RM)', 'shell%diff'))
for c in COLS:
    s, h = sol[c], she[c]
    d = 100.0 * (h - s) / s if abs(s) > 1e-30 else 0.0
    print('%-22s %16.5e %16.5e %+8.2f' % (LBL[c], s, h, d))
