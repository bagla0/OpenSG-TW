'''dehom_shell_export.py -- RM-shell dehomogenization (local STRESS + local DISP) along the
r0.2 (iea_s10) .coords, exported in the beam_extract_and_export_path_values layout as .out files.

  * coords are already in the (0,0) reference-axis frame (gen_coords_origin.py) -> NO frame shift.
  * RM bundle: build_rm_bundle(shell_yaml)  -- ref is read from the yaml's `reference` field
    (single source of truth set at yaml creation; center by default).
  * FF: center-ref RM BeamDyn per-station FF, station r0.2, VABS order [F1 F2 F3 M1 M2 M3].
  * stress frame = 'global' (section/beam frame) -> same convention as the VABS .SM that
    beam_extract_and_export_path_values.py consumes, so this .out is directly comparable.
  * STRAIN is deferred per request ("local stress and disp only for now").

export layout per path (beam_extract row-keyed):
  non_dim_path <arc-fraction 0..1>
  y2 <...>   y3 <...>                         (point coords, (0,0) frame, m)
  s_11 s_12 s_13 s_22 s_23 s_33 <...>         (local stress, Pa)
  u_1 u_2 u_3 <...>                           (local disp,  m)
'''
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm

SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')
COORDS = os.path.join(HERE, 'coords')
OUTD = os.path.join(HERE, 'out', 'dehom_shell'); os.makedirs(OUTD, exist_ok=True)
FF = np.loadtxt(os.path.join(HERE, 'beamdyn', 'ff51_rmc_reform.dat'))[10, 1:]   # r0.2, VABS order
FRAME = 'material'                    # ply-local material frame == VABS .SM ("material coordinate system")
BD_OUT = os.path.join(HERE, 'beamdyn', 'iea51rmc_bd_driver.out')  # RM BeamDyn: beam disp + rotation
BNODE = 11                            # BeamDyn output node for station 10 (r0.2): node = station + 1


def beam_kinematics(path, node):
    '''beam translation u_global + linearized rotation C (VABS frame) at a BeamDyn output node.
    matches stress_recov.py: u_local = u_global + C(w + r) - r  (C = transpose(C_Bb), small-angle).'''
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith('Time'):
            h = l.split(); row = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: row[h.index('N%03d_%s' % (node, nm))]
            TD = np.array([g('TDxr'), g('TDyr'), g('TDzr')])          # translation (BeamDyn 'r' frame)
            RD = np.array([g('RDxr'), g('RDyr'), g('RDzr')])          # Wiener-Milenkovic rotation
            u_g = np.array([TD[2], -TD[1], TD[0]])                    # -> VABS frame (B swap)
            t1, t2, t3 = RD[2], -RD[1], RD[0]                         # rotation params, VABS frame
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
            return u_g, C
    raise ValueError('no BeamDyn header in %s' % path)
NPL = 4                              # gauss points per ply through the thickness
# our Voigt stress order is [S11,S22,S33,S23,S13,S12]; beam_extract keys are 11,12,13,22,23,33
SMAP = {'11': 0, '12': 5, '13': 4, '22': 1, '23': 3, '33': 2}
PATHS = ['iea_s10.circumferential', 'iea_s10.lp_sparcap_left_thickness']

print('FF (r0.2, VABS order) F1 F2 F3 M1 M2 M3 =\n  %s' % np.array2string(FF, precision=4))
u_g, Cbeam = beam_kinematics(BD_OUT, BNODE)
print('beam kinematics r0.2 (VABS frame): u_global = %s [m]' % np.array2string(u_g, precision=4))
B = dehom_rm.build_rm_bundle(SHELL)                # ref auto-read from the yaml (reference field)
print('RM bundle ref -> frac = %s   Timo diag EA/GA2/GA3/GJ/EI2/EI3 = %s'
      % (B['frac'], np.array2string(np.diag(B['Timo']), precision=3)))


def row_line(key, vals):
    return '%s %s\n' % (key, ' '.join('%.9e' % v for v in vals))


for stem in PATHS:
    c = np.loadtxt(os.path.join(COORDS, stem + '.coords'))[:, :2]        # (0,0) frame, no shift
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))]
    nd = arc / arc[-1] if arc[-1] > 0 else arc                          # non_dim_path 0..1
    R = dehom_rm.stress_at_points(B, c, beam_force_vabs=FF, frame=FRAME, n_per_layer=NPL)
    S = np.asarray(R['stress'])                                         # [N,6] Voigt
    W = np.asarray(dehom_rm.disp_at_points(B, c, beam_force_vabs=FF))   # [N,3] warping fluctuation
    r3 = np.column_stack([np.zeros(len(c)), c[:, 0], c[:, 1]])          # section position (0, x2, x3)
    U = u_g + (Cbeam @ (W + r3).T).T - r3                              # total recovered local disp (== VABS .U)
    out = os.path.join(OUTD, stem + '.out')
    with open(out, 'w') as f:
        f.write('# RM-shell dehom  local stress [Pa, %s frame] + TOTAL recovered local disp [m]  station r0.2\n' % FRAME)
        f.write('# disp = warping(2-step) + beam disp/rotation (RM BeamDyn) : u = u_g + C(w+r) - r  == VABS .U\n')
        f.write('# VABS-order FF; (0,0) reference-axis coords; beam_extract row-keyed layout\n')
        f.write(row_line('non_dim_path', nd))
        f.write(row_line('y2', c[:, 0])); f.write(row_line('y3', c[:, 1]))
        for k in ('11', '12', '13', '22', '23', '33'):
            f.write(row_line('s_%s' % k, S[:, SMAP[k]]))
        for j, k in enumerate(('1', '2', '3')):
            f.write(row_line('u_%s' % k, U[:, j]))
    print('\n%s  (%d pts)  -> %s' % (stem, len(c), os.path.relpath(out, HERE)))
    print('  S11 [%.2f, %.2f] MPa   S12 [%.2f, %.2f] MPa   S22 [%.2f, %.2f] MPa'
          % (S[:, 0].min() / 1e6, S[:, 0].max() / 1e6, S[:, 5].min() / 1e6, S[:, 5].max() / 1e6,
             S[:, 1].min() / 1e6, S[:, 1].max() / 1e6))
    print('  u1 [%.3f, %.3f] mm    u2 [%.3f, %.3f] mm    u3 [%.3f, %.3f] mm'
          % (U[:, 0].min() * 1e3, U[:, 0].max() * 1e3, U[:, 1].min() * 1e3, U[:, 1].max() * 1e3,
             U[:, 2].min() * 1e3, U[:, 2].max() * 1e3))
print('\nDONE: 2 .out files in out/dehom_shell/  (beam_extract layout; stress + disp).')
