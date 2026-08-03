'''fbd_check.py  --  static free-body-diagram check of the BeamDyn ROOT reaction against the
applied 51 flapwise point loads (1500 Pa case).

The root reaction of a cantilever is pure statics -- it is fixed by equilibrium alone and is
INDEPENDENT of the section 6x6 stiffness.  So this table verifies the surface-traction ->
point-load -> beam-load transfer (the FBD), NOT the VABS/OpenSG cross-section homogenization.

    RootFx_analytical = sum_i Fx_i                    vs  RootFxr
    RootMy_analytical = sum_i Fx_i * z_i (z_i=eta_i*L) vs  RootMyr
    RootMz_analytical = sum_i Mz_i                     vs  RootMzr

The analytical values use the UNDEFORMED geometry.  A sub-percent bending gap is the expected
geometric arm-shortening at large tip deflection; the larger torsion gap is the fixed-direction
flap load acting through the edgewise tip deflection (a genuine geometrically-exact effect), not
a load-transfer error.
'''
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRV = os.path.join(HERE, 'iea_bd_driver.inp')
OUT = os.path.join(HERE, 'iea_bd_driver.out')
SUM = os.path.join(HERE, 'iea_bd_primary.inp.sum')

# --- applied point loads (driver table): eta Fx Fy Fz Mx My Mz ---
appl = []
for l in open(DRV).read().splitlines():
    s = l.split()
    if len(s) == 7:
        try:
            appl.append([float(x) for x in s])
        except ValueError:
            pass
appl = np.array(appl)
eta, Fx, Mz = appl[:, 0], appl[:, 1], appl[:, 6]

# --- blade length L (last node Z in the .sum) ---
sl = open(SUM).read().splitlines()
zi = next(i for i, l in enumerate(sl) if 'Initial position vectors' in l)
Z = []
for l in sl[zi + 4:]:
    s = l.split()
    if len(s) >= 5 and s[0].isdigit():
        Z.append(float(s[4]))
    elif not s:
        break
L = Z[-1]
z = eta * L

RootFx_a = Fx.sum()
RootMy_a = (Fx * z).sum()
RootMz_a = Mz.sum()

# --- BeamDyn root reactions (final steady row of the .out) ---
lines = open(OUT).read().splitlines()
hi = next(i for i, l in enumerate(lines) if l.split()[:1] == ['Time'])
names = lines[hi].split()
last = [l for l in lines[hi + 2:] if l.strip()][-1].split()
val = {n: float(v) for n, v in zip(names, last)}

rows = [('Force   RootFx [N]  ', RootFx_a, val['RootFxr']),
        ('Bending RootMy [N-m]', RootMy_a, val['RootMyr']),
        ('Torsion RootMz [N-m]', RootMz_a, val['RootMzr'])]

print('FBD / STATIC-EQUILIBRIUM CHECK  (p = 1500 Pa; L = %.3f m; %d point loads)' % (L, len(eta)))
print('root reaction is pure statics (independent of the section 6x6) -> verifies the load transfer')
print('-' * 74)
print('%-22s %16s %16s %10s' % ('quantity', 'analytical', 'BeamDyn', '%err'))
print('-' * 74)
for name, a, b in rows:
    err = 100.0 * (b - a) / a
    print('%-22s %16.6e %16.6e %+9.3f%%' % (name, a, b, err))
print('-' * 74)
