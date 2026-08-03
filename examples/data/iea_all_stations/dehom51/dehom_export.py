'''dehom_export.py -- RM-shell dehomogenization on the r0.2 path .coords, exported in the
"beam_extract_and_exportpath_values" layout, plus an immediate S11/S12 check vs the VABS .SM.

export columns (per path point):
  y2  y3  arc[m]   u1 u2 u3   S11 S22 S33 S23 S13 S12 [Pa]   e11 e22 e33 e23 e13 e12
coords/FF/.SM are in the VABS (LE) frame; the RM shell ring is reference-axis, so the coords are
shifted x -= DX before projection (DX from coords/frame_shift.dat).  FF = RM BeamDyn per-station FF.
'''
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm                                                    # sets up its own MITC/REPO paths
from scipy.spatial import cKDTree

SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')
COORDS = os.path.join(HERE, 'coords')
SM = os.path.join(ROOT, 'dehom_iea/sg_v201/iea_s10.sg.SM')
DX = float(np.loadtxt(os.path.join(COORDS, 'frame_shift.dat')))
FF = np.loadtxt(os.path.join(HERE, 'ff51_shell.dat'))[10, 1:]     # r0.2 RM FF, VABS order
COMP = ['S11', 'S22', 'S33', 'S23', 'S13', 'S12']


def export_path_values(B, coords_LE, FF, out):
    '''RM two-step dehom at coords (LE frame); shift to ref-axis, evaluate, write the export file.'''
    cref = coords_LE.copy(); cref[:, 0] -= DX                     # LE -> reference-axis for RM
    R = dehom_rm.stress_at_points(B, cref, beam_force_vabs=FF, frame='material', n_per_layer=4)
    S = np.asarray(R['stress']); E = np.asarray(R['strain'])
    U = np.asarray(dehom_rm.disp_at_points(B, cref, beam_force_vabs=FF))
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords_LE[:, 0]), np.diff(coords_LE[:, 1])))]
    M = np.column_stack([coords_LE, arc, U, S, E])
    np.savetxt(out, M, fmt='%.9e',
               header='y2 y3 arc[m]  u1 u2 u3  S11 S22 S33 S23 S13 S12[Pa]  e11 e22 e33 e23 e13 e12'
                      '   (RM shell dehom; VABS-order FF; LE-frame coords)')
    return S, E, U


print('build RM bundle from', os.path.basename(SHELL))
B = dehom_rm.build_rm_bundle(SHELL, ref='center')          # center-ref = default (matches center-ref 1D yaml)
print('RM Timo diag EA/GA2/GA3/GJ/EI2/EI3 = %s' % np.array2string(np.diag(B['Timo']), precision=3))

d = np.loadtxt(SM, skiprows=2)
smxy = d[:, :2]; sms = d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]           # -> [S11,S22,S33,S23,S13,S12]
tree = cKDTree(smxy)

for name in ('iea_s10.circumferential', 'iea_s10.lp_sparcap_left_thickness'):
    c = np.loadtxt(os.path.join(COORDS, name + '.coords'))[:, :2]
    out = os.path.join(HERE, 'rm_' + name + '.dat')
    S, E, U = export_path_values(B, c, FF, out)
    dist, idx = tree.query(c); V = sms[idx]
    e_s11 = np.linalg.norm(S[:, 0] - V[:, 0]) / (np.linalg.norm(V[:, 0]) + 1e-30) * 100
    e_s12 = np.linalg.norm(S[:, 5] - V[:, 5]) / (np.linalg.norm(V[:, 5]) + 1e-30) * 100
    print('\n%s  (%d pts, VABS-match dist max %.2e m)' % (name, len(c), dist.max()))
    print('  wrote %s' % os.path.basename(out))
    print('  RM  S11 [%.2f, %.2f] MPa   VABS S11 [%.2f, %.2f]   ->  S11 err %.1f%%'
          % (S[:, 0].min() / 1e6, S[:, 0].max() / 1e6, V[:, 0].min() / 1e6, V[:, 0].max() / 1e6, e_s11))
    print('  RM  S12 [%.2f, %.2f] MPa   VABS S12 [%.2f, %.2f]   ->  S12 err %.1f%%'
          % (S[:, 5].min() / 1e6, S[:, 5].max() / 1e6, V[:, 5].min() / 1e6, V[:, 5].max() / 1e6, e_s12))
