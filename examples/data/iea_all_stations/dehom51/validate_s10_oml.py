'''validate_s10_oml.py -- three checks before the OML dehom is trusted:
(1) GEOMETRY: the new 1-D contour must lie ON the OML = the outer envelope of the
    VABS .sg solid mesh (the old center yaml sits inset by t/2).
(2) REFERENCE: build_rm_bundle must auto-read "oml" (frac = 0).
(3) STIFFNESS: RM-OML Timoshenko diagonal vs the VABS .K of the same station.
'''
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, '..'))
XS_CANDS = [os.path.expanduser('~/OpenSG-TW-claude/examples/TW-paper/xsec_paper'),
            r'Y:\OpenSG-TW-claude\examples\TW-paper\xsec_paper']
XS = next(c for c in XS_CANDS if os.path.isdir(c))
REPO = os.path.abspath(os.path.join(XS, '..', '..', '..'))
for q in (XS, REPO, os.path.join(REPO, 'mitc_rm_segment')):
    sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import yaml
import jax

jax.config.update('jax_enable_x64', True)
import dehom_rm

OML_Y = os.path.join(IEA, 'shell51', '1d_yaml_oml', 'iea_s10_shell.yaml')
CEN_Y = os.path.join(IEA, 'shell51', '1d_yaml', 'iea_s10_shell.yaml')
VABS = os.path.join(HERE, 'out', 'VABS_iea51')


def contour_nodes(path):
    d = yaml.safe_load(open(path))
    return np.array([[float(x) for x in (r if isinstance(r, str) else r[0]).split()][:2]
                     for r in d['nodes']])


# ---- (1) geometry: contour vs solid outer envelope ----
U = np.loadtxt(os.path.join(VABS, 'iea_s10.sg.U'))
sxy = U[np.argsort(U[:, 0])][:, 1:3]
no = contour_nodes(OML_Y)
nc = contour_nodes(CEN_Y)
print('solid .sg bbox : x2 [%.4f, %.4f]  x3 [%.4f, %.4f]'
      % (sxy[:, 0].min(), sxy[:, 0].max(), sxy[:, 1].min(), sxy[:, 1].max()))
print('OML yaml bbox  : x2 [%.4f, %.4f]  x3 [%.4f, %.4f]'
      % (no[:, 0].min(), no[:, 0].max(), no[:, 1].min(), no[:, 1].max()))
print('center yaml bbox: x2 [%.4f, %.4f]  x3 [%.4f, %.4f]'
      % (nc[:, 0].min(), nc[:, 0].max(), nc[:, 1].min(), nc[:, 1].max()))
# distance from each OML contour node to the nearest solid node: should be ~0
from scipy.spatial import cKDTree

t = cKDTree(sxy)
d_oml, _ = t.query(no)
d_cen, _ = t.query(nc)
print('contour->solid nearest-node dist  OML: mean %.2f mm  max %.2f mm'
      % (d_oml.mean() * 1e3, d_oml.max() * 1e3))
print('                               center: mean %.2f mm  max %.2f mm'
      % (d_cen.mean() * 1e3, d_cen.max() * 1e3))

# ---- (2)+(3) bundle reference + stiffness vs VABS .K ----
B = dehom_rm.build_rm_bundle(OML_Y)
print('\nbundle frac =', B.get('frac'), ' ref read from yaml (0.0 = OML)')


def read_K(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if 'Timoshenko Stiffness' in l)
    j = i + 1
    while not lines[j].strip() or not lines[j].split()[0].lstrip('-').replace('.', '', 1)[0].isdigit():
        j += 1
    return np.array([[float(x) for x in lines[j + r].split()] for r in range(6)])


K = read_K(os.path.join(VABS, 'iea_s10.sg.K'))
LBL = ['EA ', 'GA2', 'GA3', 'GJ ', 'EI2', 'EI3']
print('RM-OML vs VABS .K diagonal:')
for i in range(6):
    e = 100.0 * (B['Timo'][i, i] - K[i, i]) / K[i, i]
    print('  %s  VABS %.4e   RM %.4e   %+6.2f%%' % (LBL[i], K[i, i], B['Timo'][i, i], e))
