'''vabs_export.py -- VABS counterpart of dehom_shell_export.py.  For each r0.2 .coords path, look up
the VABS local stress (.SM) and local disp (.U) at the nearest VABS point and write a .out file in
the SAME beam_extract row-keyed layout, so shell-vs-VABS is a direct file-to-file comparison.

VABS files (all at the (0,0) origin, station s10) in out/VABS_iea51/:
  iea_s10.sg.SM : "material coordinate system" stresses ; skiprows=2 ; cols  x y  s11 s12 s13 s22 s23 s33
  iea_s10.sg.U  : local warping disp                    ; cols  id x y  u1 u2 u3
'''
import os
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
COORDS = os.path.join(HERE, 'coords')
VABS = os.path.join(HERE, 'out', 'VABS_iea51')
OUTD = os.path.join(HERE, 'out', 'dehom_vabs'); os.makedirs(OUTD, exist_ok=True)
STA = 10                                                               # r0.2
PATHS = ['iea_s10.circumferential', 'iea_s10.lp_sparcap_left_thickness']

SM = np.loadtxt(os.path.join(VABS, 'iea_s%02d.sg.SM' % STA), skiprows=2)   # x y s11 s12 s13 s22 s23 s33
U = np.loadtxt(os.path.join(VABS, 'iea_s%02d.sg.U' % STA))                 # id x y u1 u2 u3
smxy = SM[:, :2]; sms = SM[:, 2:8]                                    # already beam_extract order 11,12,13,22,23,33
uxy = U[:, 1:3]; uv = U[:, 3:6]
tsm = cKDTree(smxy); tu = cKDTree(uxy)
print('VABS s%02d : .SM %d gauss pts, .U %d nodes  (material frame)' % (STA, len(SM), len(U)))


def row_line(key, vals):
    return '%s %s\n' % (key, ' '.join('%.9e' % v for v in vals))


for stem in PATHS:
    c = np.loadtxt(os.path.join(COORDS, stem + '.coords'))[:, :2]     # (0,0) frame
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))]
    nd = arc / arc[-1] if arc[-1] > 0 else arc
    dS, iS = tsm.query(c); S = sms[iS]                               # nearest VABS gauss stress
    assert dS.max() < 1e-5, '%s: VABS .SM gauss match %.2e m > 1e-5 (coords not ON the gauss points!)' % (stem, dS.max())
    # .U is NODAL, so the exact (y2,y3) rarely lands on a node -> interpolate the disp to the EXACT point
    # via inverse-distance from the 4 nearest nodes (== beam_extract getInterpPtVal); this evaluates the
    # VABS disp AT the query point, matching the RM disp location, instead of snapping to a node ~mm away.
    dU, iU = tu.query(c, k=4)
    wv = 1.0 / (dU + 1e-8 * (dU.sum(1, keepdims=True) + 1e-30)); wv /= wv.sum(1, keepdims=True)
    Uu = np.einsum('pk,pkj->pj', wv, uv[iU])                        # [P,3] disp interpolated at the point
    out = os.path.join(OUTD, stem + '.out')
    with open(out, 'w') as f:
        f.write('# VABS local stress [Pa, material frame] + local disp [m]   station r0.2 (iea_s10)\n')
        f.write('# nearest-point lookup from iea_s10.sg.SM / .U ; (0,0) coords ; beam_extract row-keyed layout\n')
        f.write(row_line('non_dim_path', nd))
        f.write(row_line('y2', c[:, 0])); f.write(row_line('y3', c[:, 1]))
        for j, k in enumerate(('11', '12', '13', '22', '23', '33')):
            f.write(row_line('s_%s' % k, S[:, j]))
        for j, k in enumerate(('1', '2', '3')):
            f.write(row_line('u_%s' % k, Uu[:, j]))
    print('\n%s  (%d pts, .SM gauss match max %.2e m [<1e-5 exact], .U 4-node interp, nearest node %.2e m)  -> %s'
          % (stem, len(c), dS.max(), dU[:, 0].max(), os.path.relpath(out, HERE)))
    print('  S11 [%.2f, %.2f] MPa   S12 [%.2f, %.2f] MPa   S22 [%.2f, %.2f] MPa'
          % (S[:, 0].min() / 1e6, S[:, 0].max() / 1e6, S[:, 1].min() / 1e6, S[:, 1].max() / 1e6,
             S[:, 3].min() / 1e6, S[:, 3].max() / 1e6))
    print('  u1 [%.3f, %.3f] mm    u2 [%.3f, %.3f] mm    u3 [%.3f, %.3f] mm'
          % (Uu[:, 0].min() * 1e3, Uu[:, 0].max() * 1e3, Uu[:, 1].min() * 1e3, Uu[:, 1].max() * 1e3,
             Uu[:, 2].min() * 1e3, Uu[:, 2].max() * 1e3))
print('\nDONE: VABS .out files in out/dehom_vabs/  (beam_extract layout).')
