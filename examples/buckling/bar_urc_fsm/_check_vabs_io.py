"""Self-check for the vabs_io .in/.SM/.U/.K readers against BAR-URC station 15."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vabs_io

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'dehom_st15')
P = os.path.join(D, 'bar_urc-15-t-0.in')

m = vabs_io.read_vabs_in(P)
print('.in   nnode=%d nelem=%d nmate=%d nlayer=%d  fmt=%d' %
      (m['nnode'], m['nelem'], m['nmate'], m['nlayer'], m['format_flag']))
print('      elem types (nodes/elem):', dict(zip(*np.unique(m['nnpe'], return_counts=True))))
print('      node1 =', m['nodes'][0], ' elem1 conn =', m['conn'][0][:4])
print('      elem1 layer=%d -> mat=%d theta1=%.3f theta3=%.4f' %
      (m['elem_layer'][0], m['elem_mat'][0], m['elem_theta1'][0], m['elem_theta3'][0]))
print('      elem7513 layer=%d -> mat=%d theta3=%.4f' %
      (m['elem_layer'][-1], m['elem_mat'][-1], m['elem_theta3'][-1]))
print('      mat1 =', m['materials'][1])
print('      mat3 =', m['materials'][3])
print('      loops: %d, sizes %s' % (len(m['loops']), [len(l) for l in m['loops'][:8]]))
print('      contour bbox y2 [%.4f %.4f]  y3 [%.4f %.4f]' %
      (m['contour'][:, 0].min(), m['contour'][:, 0].max(),
       m['contour'][:, 1].min(), m['contour'][:, 1].max()))

for ext in ('SM', 'S', 'EM', 'E', 'SMN', 'SN', 'EN', 'EMN'):
    r = vabs_io.read_sm(P + '.' + ext)
    print('.%-4s rows=%6d nodal=%-5s per-elem=%.2f  xy0=(%.6f, %.6f)  c11=%+.6e' %
          (ext, len(r['comp']), r['nodal'], len(r['comp']) / m['nelem'],
           r['xy'][0, 0], r['xy'][0, 1], r['comp'][0, 0]))

# order check: 'voigt' must equal the hard-coded reorder used in dehom_st15_figs.load_sm
raw = np.loadtxt(P + '.SM')
ref = raw[:, 2:8][:, [0, 3, 5, 4, 2, 1]]
got = vabs_io.read_sm(P + '.SM', order='voigt')['comp']
print('voigt reorder matches xsec_paper load_sm:', np.array_equal(ref, got))

u = vabs_io.read_u(P + '.U')
print('.U    rows=%d  ids 1..N=%s  matches .in coords: %s' %
      (len(u['u']), np.array_equal(u['node'], np.arange(1, m['nnode'] + 1)),
       np.allclose(u['xy'], m['nodes'])))
print('      u[node1] =', u['u'][0], ' max|U| =', np.abs(u['u']).max(axis=0))

e = vabs_io.read_ele(P + '.ELE')
print('.ELE  rows=%d (nelem=%d)  stress_m[0] c11=%+.6e' %
      (len(e['elem']), m['nelem'], e['stress_m'][0, 0]))

k = vabs_io.read_k(P + '.K')
print('.K    timo6 shape', k['timo6'].shape, ' classical4', k['classical4'].shape)
print('      area=%.6e  EA=%.6e  mass/span=%.6e' % (k['area'], k['EA'], k['mass_per_span']))
print('      GJ_classical=%.10E   GJ_timo(K44)=%.10E   ratio=%.4f' %
      (k['GJ_classical'], k['GJ_timo'], k['GJ_timo'] / k['GJ_classical']))
print('      shear_center=', k['shear_center'], ' tension_center=', k['tension_center'])
print('      angles:', k['angles'])
print('      symmetry: timo6 %.2e  classical4 %.2e' %
      (np.abs(k['timo6'] - k['timo6'].T).max(), np.abs(k['classical4'] - k['classical4'].T).max()))
print('      timo6 @ timo_compliance6 == I:',
      np.allclose(k['timo6'] @ k['timo_compliance6'], np.eye(6), atol=1e-8))
np.set_printoptions(precision=6, linewidth=200)
print('\nStation-15 Timoshenko 6x6 (ext, shear2, shear3, twist, bend5, bend6):')
print(k['timo6'])
