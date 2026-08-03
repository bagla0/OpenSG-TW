'''debug the thickness-path stress misalignment: per point, show the RM projected element, through-
thickness depth z, the shell laminate at that element (name + total thickness + ply stack), the RM
sigma11, and the VABS .SM sigma11 -- to see whether the wall thickness / ply ordering / z-sign is off.'''
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm

SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')
c = np.loadtxt(os.path.join(HERE, 'coords', 'iea_s10.lp_sparcap_left_thickness.coords'))[:, :2]
FF = np.loadtxt(os.path.join(HERE, 'beamdyn', 'ff51_rmc_reform.dat'))[10, 1:]
SM = np.loadtxt(os.path.join(HERE, 'out', 'VABS_iea51', 'iea_s10.sg.SM'), skiprows=2)
tsm = cKDTree(SM[:, :2])

B = dehom_rm.build_rm_bundle(SHELL, ref='center')
R = dehom_rm.stress_at_points(B, c, beam_force_vabs=FF, frame='material', n_per_layer=4)
S = np.asarray(R['stress']); el = R['elem'].astype(int); z = R['depth']; proj = R['proj']
lp = B['layup_per_elem']; ldb = B['layup_db']
_, iV = tsm.query(c); Vs11 = SM[iV, 2] / 1e6                         # VABS s11 (col 2), MPa

print('shell 1D yaml: %d ring elements' % len(lp))
# total wall thickness of each DISTINCT layup that the path hits
hit = sorted(set(el))
for e in hit:
    ln = lp[e]; th = ldb[ln]['thick']; ang = ldb[ln].get('angles'); mats = ldb[ln].get('mat_names')
    print('  elem %3d  layup=%-16s  n_ply=%d  total_t=%.4f m  mats=%s' % (e, ln, len(th), sum(th), mats))

print('\n idx     y3       arc[mm]   elem  depth_z[mm]   RM_s11    VABS_s11   ply/thk')
arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))] * 1e3
for i in range(len(c)):
    e = el[i]; tt = sum(ldb[lp[e]]['thick'])
    print('  %2d  %8.4f  %8.2f   %4d   %8.2f     %8.2f  %8.2f    t_wall=%.1fmm z/t=%+.2f'
          % (i, c[i, 1], arc[i], e, z[i] * 1e3, S[i, 0] / 1e6, Vs11[i], tt * 1e3, z[i] / (0.5 * tt)))
