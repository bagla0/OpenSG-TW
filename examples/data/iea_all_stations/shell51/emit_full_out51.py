'''emit_full_out51.py -- write ONE full .out per station per source (RM shell / JAX solid), holding
the Timoshenko 6x6 stiffness [K] + compliance [S]=inv(K) AND the FULL VABS-.K sectional-properties
block (mass 6x6, centers, classical, principal, mass properties) appended after the Compliance, in
VABS .K print order.  Each source gets its OWN mass matrix (shell FAITHFUL mass / solid area mass)
and its OWN centers computed from its own K and M.  Everything referenced to the x1 reference axis.

    python emit_full_out51.py --source rm       # -> out/OpenSG_RM_Shell/*.out
    python emit_full_out51.py --source jax      # -> out/OpenSG_JAX_Solid/*.out
'''
import argparse
import glob
import os
import re

import numpy as np

import sectional_props as sp

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
WINDIO = os.path.join(BASE, 'IEA-22-280-RWT.yaml')

SRC = {
    'rm':  dict(homo='homo_rm',  prefix='OpenSG_RM_',  suffix='OpenSG_RM_Shell',
                y='1d_yaml', ysuf='_shell.yaml', log='log_rm.txt',
                label='Timoshenko 6x6 -- RM SHELL cross-section (OpenSG-RM, Boundary ring, MITC gamma_23)'),
    'jax': dict(homo='homo_jax', prefix='OpenSG_JAX_', suffix='OpenSG_JAX_Solid',
                y='2d_yaml', ysuf='_solid.yaml', log='log_jax.txt',
                label='Timoshenko 6x6 -- 2-D SOLID cross-section (OpenSG-JAX, JAX-FEM Mechanics of Structure Genome)'),
}


def load_times(path):
    t = {}
    if not os.path.exists(path):
        return t
    pat = re.compile(r'^\[\s*(\S+)\s*\].*\[([\d.]+)s\]\s*$')
    for ln in open(path):
        m = pat.match(ln.strip())
        if m:
            try:
                t[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return t


# ---------------------------------------------------------------- formatters
def m6(f, title, M):
    f.write(' %s\n ========================================================\n\n' % title)
    for i in range(6):
        f.write('   ' + '  '.join('% .12e' % M[i, j] for j in range(6)) + '\n')
    f.write('\n')


def m4(f, title, M):
    f.write(' %s\n ========================================================\n\n' % title)
    for i in range(4):
        f.write('   ' + '  '.join('% .12e' % M[i, j] for j in range(4)) + '\n')
    f.write('\n')


def vec2(f, title, v):
    f.write(' %s\n ========================================================\n\n' % title)
    f.write('   % .12e  % .12e\n\n' % (v[0], v[1]))


def scal(f, txt, val):
    f.write(' %-38s = % .10E\n' % (txt, val))


def write_block(f, c):
    K = c['K']; S = c['S']; M = c['M']
    # --- Mass 6x6 (at reference axis)
    m6(f, 'The 6X6 Mass Matrix', M)
    vec2(f, 'The Mass Center of the Cross Section', c['mass_center'])
    m6(f, 'The 6X6 Mass Matrix at the Mass Center', c['Mc'])
    f.write(' The Mass Properties with respect to Principal Inertial Axes\n'
            ' ========================================================\n\n')
    scal(f, 'Mass per unit span', c['mu'])
    scal(f, 'Mass moment of inertia i11', c['i11'])
    scal(f, 'Principal mass moments of inertia i22', c['i22'])
    scal(f, 'Principal mass moments of inertia i33', c['i33'])
    f.write(' The principal inertial axes rotated from user coordinate system by\n'
            '   %.10f degrees about the positive direction of x1 axis.\n' % c['ang_I'])
    scal(f, 'The mass-weighted radius of gyration', c['rg'])
    f.write('\n')
    vec2(f, 'The Geometric Center of the Cross Section', c['geom_center'])
    f.write(' The Area of the Cross Section\n ========================================================\n\n')
    f.write(' Area = % .10E\n\n' % c['area'])
    m4(f, 'Classical Stiffness Matrix (1-extension; 2-twist; 3,4-bending)', c['ClsStiff'])
    m4(f, 'Classical Compliance Matrix (1-extension; 2-twist; 3,4-bending)', c['ClsComp'])
    vec2(f, 'The Tension Center of the Cross Section', c['tension_center'])
    scal(f, 'The extension stiffness EA', c['EA'])
    scal(f, 'The torsional stiffness GJ', c['GJ'])
    scal(f, 'Principal bending stiffness EI22', c['EI22'])
    scal(f, 'Principal bending stiffness EI33', c['EI33'])
    f.write(' The principal bending axes rotated from the user coordinate system by\n'
            '   %.10f degrees about the positive direction of x1 axis.\n\n' % c['ang_EI'])
    m6(f, 'Timoshenko Stiffness Matrix (1-extension; 2,3-shear, 4-twist; 5,6-bending)', K)
    m6(f, 'Timoshenko Compliance Matrix (1-extension; 2,3-shear, 4-twist; 5,6-bending)', S)
    vec2(f, 'The Shear Center of the Cross Section', c['shear_center'])
    scal(f, 'Principal shear stiffness GA22', c['GA22'])
    scal(f, 'Principal shear stiffness GA33', c['GA33'])
    f.write(' The principal shear axes rotated from user coordinate system by\n'
            '   %.10f degrees about the positive direction of x1 axis.\n\n' % c['ang_GA'])
    m4(f, 'Classical Stiffness Matrix at Shear Center (1-extension; 2-twist; 3,4-bending)', c['ClsStiffSC'])
    m4(f, 'Classical Compliance Matrix at Shear Center (1-extension; 2-twist; 3,4-bending)', c['ClsCompSC'])
    m6(f, 'The Mass Matrix at Shear Center', c['MassSC'])
    vec2(f, 'The Mass Center with respect to Shear Center', c['mc_wrt_sc'])
    f.write(' The Mass Properties at Shear Center\n'
            ' ========================================================\n\n')
    sc = c['scmass']
    scal(f, 'Mass per unit span', sc['mu'])
    scal(f, 'Mass moment of inertia i11', sc['i11'])
    scal(f, 'Principal mass moments of inertia i22', sc['i22'])
    scal(f, 'Principal mass moments of inertia i33', sc['i33'])
    f.write(' The principal inertial axes rotated from the user coordinate system by\n'
            '   %.10f degrees about the positive direction of x1 axis.\n' % sc['ang'])
    scal(f, 'The mass-weighted radius of gyration', sc['rg'])
    f.write('\n')


def head_mat(f, name, M):
    f.write('%s:\n' % name)
    for i in range(6):
        f.write('  ' + '  '.join('% .10e' % M[i, j] for j in range(6)) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=['rm', 'jax'])
    ap.add_argument('--ydir', default=None, help='override the mesh folder (e.g. pynumad_quad)')
    ap.add_argument('--outsuffix', default=None, help='override the out/ subfolder name')
    a = ap.parse_args()
    s = SRC[a.source]
    homo = os.path.join(HERE, s['homo'])
    ydir = os.path.join(HERE, a.ydir) if a.ydir else os.path.join(HERE, s['y'])
    if a.outsuffix:
        s = dict(s, suffix=a.outsuffix)
    outdir = os.path.join(HERE, 'out', s['suffix'])
    os.makedirs(outdir, exist_ok=True)
    times = load_times(os.path.join(HERE, s['log']))
    files = sorted(glob.glob(os.path.join(homo, s['prefix'] + '*.txt')))
    n = 0; miss_all = set()
    for fp in files:
        tag = os.path.basename(fp).replace(s['prefix'], '').replace('.txt', '')  # iea_sNN
        K = np.loadtxt(fp)
        try:
            S = np.linalg.inv(K)
        except Exception:
            S = np.full((6, 6), np.nan)
        yp = os.path.join(ydir, tag + s['ysuf'])
        if not os.path.exists(yp):
            print('SKIP %s (no %s)' % (tag, yp)); continue
        if a.source == 'rm':
            M, area, gc = sp.shell_mass_and_geom(yp)
        else:
            M, area, gc, miss = sp.solid_mass_and_geom(yp, WINDIO)
            miss_all |= miss
        c = sp.compute_props(K, M, area, gc)
        tt = times.get(tag)
        with open(os.path.join(outdir, '%s_%s.out' % (tag, s['suffix'])), 'w') as f:
            f.write('# %s (reference-axis origin x1)\n' % s['label'])
            f.write('# convention (VABS/OpenSG order): 1=extension, 2-3=transverse shear, '
                    '4=torsion, 5-6=bending\n')
            f.write('# origin = windIO reference axis (x1); Time-taken: %s\n\n'
                    % (('%.2f s' % tt) if tt is not None else 'n/a'))
            head_mat(f, 'Stiffness ', K)
            f.write('\n')
            head_mat(f, 'Compliance', S)
            f.write('\n# ===== Sectional properties (VABS .K order, referenced to x1) =====\n\n')
            write_block(f, c)
        n += 1
    print('wrote %d x *_%s.out -> %s' % (n, s['suffix'], outdir))
    if miss_all:
        print('WARNING missing-density materials:', miss_all)


if __name__ == '__main__':
    main()
