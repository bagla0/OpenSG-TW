'''debug the abrupt jump(s) on the circumferential path: locate the worst RM-vs-VABS points and show
their projected ring element, through-thickness depth, layup + ply, so we can see whether the surface
point is snapping/projecting to a deep (carbon) z at a web/cap junction.'''
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
c = np.loadtxt(os.path.join(HERE, 'coords', 'iea_s10.circumferential.coords'))[:, :2]
FF = np.loadtxt(os.path.join(HERE, 'beamdyn', 'ff51_rmc_reform.dat'))[10, 1:]
SM = np.loadtxt(os.path.join(HERE, 'out', 'VABS_iea51', 'iea_s10.sg.SM'), skiprows=2)
tsm = cKDTree(SM[:, :2])

B = dehom_rm.build_rm_bundle(SHELL)
R = dehom_rm.stress_at_points(B, c, beam_force_vabs=FF, frame='material', n_per_layer=4)
S = np.asarray(R['stress']); el = R['elem'].astype(int); z = R['depth']; proj = R['proj']
lp = B['layup_per_elem']; ldb = B['layup_db']
dist, iV = tsm.query(c); Vs11 = SM[iV, 2] / 1e6                     # exact VABS gauss (dist~0)
arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))] * 1e3

d11 = np.abs(S[:, 0] / 1e6 - Vs11)
order = np.argsort(-d11)
print('circumferential: %d pts ; snap dist max %.2e m' % (len(c), dist.max()))
print('WORST 8 |RM-VABS| sigma11 points:')
print(' idx    arc[mm]   coord(x,y)          snapdist   elem  layup       t[mm]  z_oml[mm]  RM_s11   VABS_s11')
for i in order[:8]:
    e = el[i]; hth = sum(ldb[lp[e]]['thick']); frac = B['frac']
    zoml = (z[i] + frac * hth) * 1e3
    print('  %3d  %8.1f  (%7.3f,%7.3f)  %.2e   %4d  %-10s  %5.1f  %+8.2f  %8.2f  %8.2f'
          % (i, arc[i], c[i, 0], c[i, 1], dist[i], e, lp[e], hth * 1e3, zoml, S[i, 0] / 1e6, Vs11[i]))
