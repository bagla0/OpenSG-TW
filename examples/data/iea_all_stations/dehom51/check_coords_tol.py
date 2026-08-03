'''check_coords_tol.py -- verify RM and VABS are evaluated at IDENTICAL coordinates.
Compares y2,y3 rows of the RM .out (center: dehom_shell/, OML: dehom_shell_oml/) and
the VABS .out (dehom_vabs/) against each other and against the .coords files.
PASS = max |dy2|,|dy3| < 1e-5 m everywhere.'''
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
PATHS = ['iea_s10.circumferential', 'iea_s10.lp_sparcap_left_thickness']


def read_out(p):
    d = {}
    for ln in open(p):
        if ln.startswith('#') or not ln.strip():
            continue
        t = ln.split()
        d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


for pth in PATHS:
    co = np.loadtxt(os.path.join(HERE, 'coords', pth + '.coords'))[:, :2]
    vb = read_out(os.path.join(OUT, 'dehom_vabs', pth + '.out'))
    rm_c = read_out(os.path.join(OUT, 'dehom_shell', pth + '.out'))
    rm_o = read_out(os.path.join(OUT, 'dehom_shell_oml', pth + '.out'))
    print('== %s  (%d coords pts)' % (pth, len(co)))
    for nm, d in (('VABS   ', vb), ('RM-cen ', rm_c), ('RM-oml ', rm_o)):
        n = min(len(co), len(d['y2']))
        dy2 = np.abs(d['y2'][:n] - co[:n, 0]).max()
        dy3 = np.abs(d['y3'][:n] - co[:n, 1]).max()
        ok = 'PASS' if max(dy2, dy3) < 1e-5 else 'FAIL'
        print('   %s vs coords: pts %3d  max|dy2| %.2e  max|dy3| %.2e   %s'
              % (nm, len(d['y2']), dy2, dy3, ok))
    n = min(len(vb['y2']), len(rm_c['y2']))
    print('   VABS vs RM-cen : max|dy2| %.2e  max|dy3| %.2e'
          % (np.abs(vb['y2'][:n] - rm_c['y2'][:n]).max(),
             np.abs(vb['y3'][:n] - rm_c['y3'][:n]).max()))
