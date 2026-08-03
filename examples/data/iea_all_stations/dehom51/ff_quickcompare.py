'''quick: are the FF from the RM BeamDyn and the solid BeamDyn one-to-one across all 51 stations?
uses the already-extracted ff51_shell.dat (RM) and ff51_solid.dat (solid).'''
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rm = np.loadtxt(os.path.join(HERE, 'ff51_shell.dat'))
sol = np.loadtxt(os.path.join(HERE, 'ff51_solid.dat'))
lbl = ['F1(axial)', 'F2', 'F3(shear)', 'M1(torsion)', 'M2(flapbend)', 'M3(edgebend)']
print('FF one-to-one: RM BeamDyn vs solid BeamDyn (VABS order), all 51 stations')
print('%3s %6s  %11s %11s %11s  %8s' % ('st', 'eta', 'F3 RM', 'M2 RM', 'M1 RM', 'max%diff'))
worst = 0.0; worst_st = -1
for i in range(51):
    a, b = rm[i, 1:], sol[i, 1:]
    dom = np.abs(a) > 1e-2 * np.max(np.abs(a))
    pd = np.max(np.abs((a[dom] - b[dom]) / a[dom])) * 100 if dom.any() else 0.0
    if pd > worst:
        worst, worst_st = pd, i
    if i % 5 == 0 or pd > 1.5:
        print('%3d %6.3f  %11.4e %11.4e %11.4e  %7.2f%%' % (i, rm[i, 0], a[2], a[4], a[3], pd))
print('\nWORST dominant-component FF diff = %.2f%% at station %d (eta=%.3f)' % (worst, worst_st, rm[worst_st, 0]))
print('mean over stations = %.3f%%' % np.mean([
    np.max(np.abs((rm[i, 1:] - sol[i, 1:])[np.abs(rm[i, 1:]) > 1e-2 * np.max(np.abs(rm[i, 1:]))] /
                  rm[i, 1:][np.abs(rm[i, 1:]) > 1e-2 * np.max(np.abs(rm[i, 1:]))])) * 100 for i in range(51)]))
