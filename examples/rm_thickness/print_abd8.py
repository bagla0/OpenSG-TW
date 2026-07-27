"""print_abd8.py -- the 8x8 two-dimensional constitutive law
[[A,B,0],[B,D,0],[0,0,G]] produced by the structure gene, for the paper.

Prints, for chosen laminates, the A/B/D blocks (shared by every model) and the two
shear blocks side by side: G_MSG from the RM projection versus the FSDT
k*As with k = 5/6.  LaTeX-ready pmatrix bodies with 3 significant figures.
"""
import numpy as np

from jaxcfg import jnp                    # noqa: F401
import sg_plate as SG
import models as M
from materials import MATDB

import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'navier_plate'))
from navier_models import MATDB_MR        # noqa: E402

ALLDB = {**MATDB, **MATDB_MR}


def fmt(x):
    if x == 0 or abs(x) < 1e-30:
        return '0'
    return f"{x:.3g}".replace('e+0', '\\times10^{').replace('e-0', '\\times10^{-') \
        .replace('e+', '\\times10^{').replace('e-', '\\times10^{-') + \
        ('}' if 'times' in f"{x:.3g}".replace('e', 'times') else '')


def fmt2(x):
    """engineering: mantissa x 10^n, 3 sig figs, LaTeX."""
    if x == 0 or abs(x) < 1e-30:
        return '0'
    n = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** n
    return f"{m:.2f}\\!\\times\\!10^{{{n}}}"


def block(Mx):
    rows = [' & '.join(fmt2(v) for v in row) for row in np.asarray(Mx)]
    return ' \\\\\n'.join(rows)


def report(name, thick, angles, mats, npl=8):
    sg = SG.build(thick, angles, mats, ALLDB, n_per_layer=npl, elem_order=3)
    A6 = np.asarray(sg['A6'])
    A, B, D = A6[:3, :3], A6[:3, 3:], A6[3:, 3:]
    Gmsg = np.asarray(sg['G_msg'])
    Gfsdt = np.asarray(M.fsdt_shear(sg)[0])
    print('=' * 72)
    print(name)
    print('=' * 72)
    print('A [N/m]:\n' + block(A))
    print('B [N]:\n' + block(B))
    print('D [N m]:\n' + block(D))
    print('G_MSG [N/m]:\n' + block(Gmsg))
    print('G_FSDT = (5/6) As [N/m]:\n' + block(Gfsdt))
    r = Gmsg / Gfsdt
    print('ratio G_MSG / G_FSDT (elementwise diag):',
          f"{r[0,0]:.3f}", f"{r[1,1]:.3f}")


if __name__ == '__main__':
    report('M-R sandwich, H = 0.25 m (Example 2)',
           [0.025, 0.2, 0.025], [0., 0., 0.], ['mr_face', 'mr_core', 'mr_face'])
    report('[0/90/0] Pagano, h = 1 m (Example 1)',
           [1 / 3] * 3, [0., 90., 0.], ['pagano'] * 3, npl=6)
