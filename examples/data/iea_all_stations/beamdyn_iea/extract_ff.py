'''extract_ff.py  --  verify the static BeamDyn solve and extract the per-node internal FF
(sectional force/moment resultants) for cross-section dehomogenization.

Reads bd_driver_iea.out (channel header + final steady row) and bd_primary_iea.inp.sum
(node positions). Prints the convergence/statics checks and the per-node FF table, and also
runs beamdyn_trans.beam_reaction() as a cross-check.
'''
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from beamdyn_trans import beam_reaction

OUT = os.path.join(HERE, 'iea_bd_driver.out')
SUM = os.path.join(HERE, 'iea_bd_primary.inp.sum')
DRV = os.path.join(HERE, 'iea_bd_driver.inp')

# the 16 windIO airfoil spanwise_position etas (the cross-section stations for dehomogenization)
STATION_ETAS = np.array([0.0, 0.02, 0.04868313468735217, 0.06649308419703373,
                         0.08345591801468194, 0.10220904193570582, 0.11036272014164386,
                         0.1364246061019374, 0.15564440587515185, 0.19665336575444797,
                         0.24696148735364706, 0.3992636115637571, 0.5335887750152993,
                         0.738938689884722, 0.9799991709122947, 1.0])

# ---------------------------------------------------------------- parse .out header + final row
lines = open(OUT).read().splitlines()
hi = next(i for i, l in enumerate(lines) if l.split()[:1] == ['Time'])
names = lines[hi].split()
data = [l for l in lines[hi + 2:] if l.strip()]
last = data[-1].split()
val = {n: float(v) for n, v in zip(names, last)}

# ---------------------------------------------------------------- node Z (span) from .sum
sl = open(SUM).read().splitlines()
zi = next(i for i, l in enumerate(sl) if 'Initial position vectors' in l)
Z = []
for l in sl[zi + 4:]:
    s = l.split()
    if len(s) >= 5 and s[0].isdigit():
        Z.append(float(s[4]))
    elif not s:
        break
Z = np.array(Z)
nn = len(Z)
L = Z[-1]


def nc(base):
    return np.array([val['N%03d_%s' % (k + 1, base)] for k in range(nn)])


FxL, FyL, FzL = nc('FxL'), nc('FyL'), nc('FzL')     # section-LOCAL frame forces (dehom input)
Fxr, Fyr, Fzr = nc('Fxr'), nc('Fyr'), nc('Fzr')     # root frame forces (for the global statics check)
MxL, MyL, MzL = nc('MxL'), nc('MyL'), nc('MzL')     # section-LOCAL frame moments (dehom input)
TDx = nc('TDxr')

# ---------------------------------------------------------------- total applied load (driver table)
appl = []
for l in open(DRV).read().splitlines():
    s = l.split()
    if len(s) == 7:
        try:
            appl.append([float(x) for x in s])
        except ValueError:
            pass
appl = np.array(appl)                                   # eta Fx Fy Fz Mx My Mz  (51 rows)
sumFx = appl[:, 1].sum()
sumMz = appl[:, 6].sum()
M_arm = float((appl[:, 1] * appl[:, 0] * L).sum())      # Sum Fx_i * z_i  (expected root bending)

print('=' * 78)
print('CONVERGENCE / STATICS CHECK  (static solve, final steady row)')
print('=' * 78)
print('  applied  Sum Fx           = %14.4f N   (= %.4f MN)' % (sumFx, sumFx / 1e6))
print('  reaction RootFxr          = %14.4f N   (= %.4f MN)' % (val['RootFxr'], val['RootFxr'] / 1e6))
print('     -> ratio RootFxr/SumFx = %.6f' % (val['RootFxr'] / sumFx))
print('  applied  Sum Fx_i*z_i     = %14.4e N-m (expected root bending)' % M_arm)
print('  reaction RootMyr          = %14.4e N-m' % val['RootMyr'])
print('     -> ratio RootMyr/Sum   = %.6f' % (val['RootMyr'] / M_arm))
print('  applied  Sum Mz           = %14.4e N-m (applied torsion)' % sumMz)
print('  reaction RootMzr          = %14.4e N-m' % val['RootMzr'])
print('  node1 Fxr (root shear)    = %14.4f N   (should equal RootFxr)' % Fxr[0])
print('  node%d Fxr (tip shear)    = %14.4f N   (should ~ tip point load)' % (nn, Fxr[-1]))
print('  TIP flapwise defl TipTDxr = %14.4f m' % val['TipTDxr'])
print('  TIP edgewise  defl TipTDyr = %13.4f m' % val['TipTDyr'])
print('  TIP twist     RDzr        = %14.4e (-)' % val['TipRDzr'])

print()
print('=' * 96)
print('RAW PER-NODE INTERNAL FF  (11 Gaussian GLL nodes; section-LOCAL "_L" frame)')
print('=' * 96)
print('%3s %8s %6s %13s %13s %13s %13s %13s %13s %9s'
      % ('nd', 'z[m]', 'eta', 'FxL[N]', 'FyL[N]', 'FzL[N]', 'MxL[N-m]', 'MyL[N-m]', 'MzL[N-m]', 'TDxr[m]'))
print('%3s %8s %6s %13s %13s %13s %13s %13s %13s %9s'
      % ('', '', '', 'flapshear', 'edgeshear', 'axial', 'edgebend', 'flapbend', 'torsion', 'flapdefl'))
for k in range(nn):
    print('%3d %8.3f %6.4f %13.4e %13.4e %13.4e %13.4e %13.4e %13.4e %9.4f'
          % (k + 1, Z[k], Z[k] / L, FxL[k], FyL[k], FzL[k], MxL[k], MyL[k], MzL[k], TDx[k]))

# ---------------------------------------------------------------- cubic-interpolate onto 16 stations
eta_nd = Z / L
zst = STATION_ETAS * L


def cub(y):
    return CubicSpline(eta_nd, y)(STATION_ETAS)


sFxL, sFyL, sFzL = cub(FxL), cub(FyL), cub(FzL)
sMxL, sMyL, sMzL = cub(MxL), cub(MyL), cub(MzL)
sTDx = cub(TDx)

print()
print('=' * 96)
print('DELIVERABLE: FF at the 16 CROSS-SECTION STATIONS  (cubic interp of nodal FF; section-LOCAL frame)')
print('=' * 96)
print('%3s %8s %7s %13s %13s %13s %13s %13s %13s %9s'
      % ('st', 'z[m]', 'eta', 'FxL[N]', 'FyL[N]', 'FzL[N]', 'MxL[N-m]', 'MyL[N-m]', 'MzL[N-m]', 'TDxr[m]'))
print('%3s %8s %7s %13s %13s %13s %13s %13s %13s %9s'
      % ('', '', '', 'flapshear', 'edgeshear', 'axial', 'edgebend', 'flapbend', 'torsion', 'flapdefl'))
for k in range(len(STATION_ETAS)):
    print('%3d %8.3f %7.5f %13.4e %13.4e %13.4e %13.4e %13.4e %13.4e %9.4f'
          % (k + 1, zst[k], STATION_ETAS[k], sFxL[k], sFyL[k], sFzL[k],
             sMxL[k], sMyL[k], sMzL[k], sTDx[k]))

# save the 16-station FF for downstream dehomogenization
np.savetxt(os.path.join(HERE, 'ff_16stations.dat'),
           np.column_stack([STATION_ETAS, zst, sFxL, sFyL, sFzL, sMxL, sMyL, sMzL]),
           header='eta  z[m]  FxL[N]  FyL[N]  FzL[N]  MxL[N-m]  MyL[N-m]  MzL[N-m]  '
                  '(section-local frame; cubic interp of BeamDyn Gaussian nodal FF; '
                  'straight untwisted beam, 1200 Pa flapwise static)', fmt='%.8e')
print('\nwrote ff_16stations.dat')

# ---------------------------------------------------------------- beam_reaction cross-check
print()
print('=' * 78)
print('beam_reaction() cross-check  (label it pulled : value)')
print('=' * 78)
try:
    bf, bd = beam_reaction(OUT)
    print('num_segments returned = %d' % len(bf))
    print('root segment force 6-vector (label=value):')
    for lab, v in bf[0]:
        print('   %-10s = %14.5e' % (lab, v))
    print('tip  segment force 6-vector (label=value):')
    for lab, v in bf[-1]:
        print('   %-10s = %14.5e' % (lab, v))
except Exception as e:
    print('beam_reaction raised:', repr(e))
