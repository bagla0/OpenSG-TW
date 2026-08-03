'''extract_ff51.py -- per-station sectional forces FF (VABS order) from the 51-station TRAPEZOIDAL
BeamDyn outputs, for cross-section dehomogenization.

  RM shell dehom  <- iea51_shell_bd_driver.out   (RM Timoshenko 6x6 beam)
  solid   dehom   <- iea51_solid_bd_driver.out   (2-D solid Timoshenko 6x6 beam)

BeamDyn writes per-node internal resultants in the section-LOCAL "_L" frame:
  FxL=flap shear, FyL=edge shear, FzL=AXIAL ; MxL=edge bend, MyL=flap bend, MzL=TORSION.
The dehom code wants them in VABS order [F1,F2,F3,M1,M2,M3]=[axial,shear2,shear3,torsion,bend2,bend3]
via the BeamDyn->VABS beam-axis swap B=[[0,0,1],[0,-1,0],[1,0,0]] (same map as stress_recov.py and
the .glb work):
      FF_vabs = [ FzL, -FyL, FxL,  MzL, -MyL, MxL ]

Trapezoidal quadrature => output node k (1-based) sits AT station iea_s{k-1}, eta=(k-1)/50.
'''
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.path.join(HERE, '..', 'beamdyn_iea')
MODELS = {'shell': 'iea51_shell_bd_driver.out', 'solid': 'iea51_solid_bd_driver.out'}


def parse_last(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith('Time'):
            h = l.split()
            d = np.array([r.split() for r in L[i + 2:]], float)
            return h, d[-1]
    raise ValueError('no header in ' + path)


def ff_vabs(h, row, k):
    g = lambda n: row[h.index('N%03d_%s' % (k, n))]
    Fx, Fy, Fz = g('FxL'), g('FyL'), g('FzL')
    Mx, My, Mz = g('MxL'), g('MyL'), g('MzL')
    return [Fz, -Fy, Fx, Mz, -My, Mx]                      # VABS order [F1,F2,F3,M1,M2,M3]


for tag, fname in MODELS.items():
    h, row = parse_last(os.path.join(BD, fname))
    N = sum(1 for x in h if x.endswith('_TDxr') and x.startswith('N'))
    rows = [[(k - 1) / 50.0] + ff_vabs(h, row, k) for k in range(1, N + 1)]
    out = os.path.join(HERE, 'ff51_%s.dat' % tag)
    np.savetxt(out, np.array(rows), fmt='%.8e',
               header='eta  F1(axial)  F2(shear2)  F3(shear3)  M1(torsion)  M2(bend2)  M3(bend3)   '
                      'VABS order; BeamDyn _L frame -> B swap; source %s' % fname)
    print('wrote %s  (%d stations)' % (os.path.basename(out), N))

# quick sanity print at r0.2 (station index 10 -> node 11)
for tag in MODELS:
    ff = np.loadtxt(os.path.join(HERE, 'ff51_%s.dat' % tag))
    r = ff[10]
    print('  %-5s r0.2  FF=[%.3e %.3e %.3e %.3e %.3e %.3e]' % (tag, *r[1:]))
