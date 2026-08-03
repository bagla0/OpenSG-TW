'''check homo<->BeamDyn one-to-one, and RM vs .glb/VABS at r0.2, and the current origin/frame.'''
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm


def parse_out_K(path):
    L = open(path).read().splitlines()
    for i, l in enumerate(L):
        if l.strip().startswith('Stiffness'):
            rows = []
            j = i + 1
            while len(rows) < 6 and j < len(L):
                v = L[j].split()
                try:
                    fv = [float(x) for x in v]
                    if len(fv) >= 6:
                        rows.append(fv[:6])
                except ValueError:
                    pass
                j += 1
            return np.array(rows)
    return None


RM_OUT = os.path.join(ROOT, 'shell51/out/OpenSG_RM_Shell/iea_s10_OpenSG_RM_Shell.out')
SOL_OUT = os.path.join(ROOT, 'shell51/out/OpenSG_Hybrid_Solid/iea_s10_OpenSG_Hybrid_Solid.out')
SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')

Krm_out = parse_out_K(RM_OUT)
Ksol_out = parse_out_K(SOL_OUT)
B = dehom_rm.build_rm_bundle(SHELL, ref='oml')
Krm_bun = np.asarray(B['Timo'])

lbl = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
print('=== (G) RM homo one-to-one:  build_rm_bundle 6x6  vs  OpenSG_RM_Shell .out (BeamDyn input) ===')
print('%-5s %14s %14s %9s' % ('term', 'bundle(dehom)', 'RM .out(BeamDyn)', '%diff'))
for k in range(6):
    a, b = Krm_bun[k, k], Krm_out[k, k]
    print('%-5s %14.4e %14.4e %8.2f' % (lbl[k], a, b, 100 * (a - b) / b))
fro = np.linalg.norm(Krm_bun - Krm_out) / np.linalg.norm(Krm_out) * 100
print('  full-6x6 Frobenius diff = %.2f%%   %s' % (fro, 'ONE-TO-ONE' if fro < 1 else '<-- MISMATCH!'))

print('\n=== solid 6x6 (OpenSG_Hybrid_Solid .out = solid BeamDyn input, drives the .glb) ===')
print('  diag = %s' % np.array2string(np.diag(Ksol_out), precision=3))
print('  RM/solid diag ratio = %s' % np.array2string(np.diag(Krm_bun) / np.diag(Ksol_out), precision=3))

print('\n=== FF one-to-one (r0.2, VABS order) ===')
ffs = np.loadtxt(os.path.join(HERE, 'ff51_shell.dat'))[10, 1:]
ffo = np.loadtxt(os.path.join(HERE, 'ff51_solid.dat'))[10, 1:]
print('  RM  BeamDyn FF = %s' % np.array2string(ffs, precision=3))
print('  sol BeamDyn FF = %s' % np.array2string(ffo, precision=3))
glb = os.path.join(ROOT, 'dehom_iea/glb51/iea_s10.sg.glb')
g = [l.split() for l in open(glb).read().splitlines() if l.strip()]
F1, M1, M2, M3 = [float(x) for x in g[4]]; F2, F3 = [float(x) for x in g[5]]
print('  .glb FF (F1 F2 F3 M1 M2 M3) = [%.3e %.3e %.3e %.3e %.3e %.3e]' % (F1, F2, F3, M1, M2, M3))
print('  -> .glb == solid BeamDyn FF ?  %s' % np.allclose([F1, F2, F3, M1, M2, M3], ffo, rtol=1e-3))

print('\n=== origin / frame of current sg + SM/U/EM ===')
for f, lab in [('shell51/sg_v201/iea_s10.sg', '.sg'),
               ('dehom_iea/sg_v201/iea_s10.sg.SM', '.SM'),
               ('dehom_iea/sg_v201/iea_s10.sg.U', '.U'),
               ('dehom_iea/sg_v201/iea_s10.sg.EM', '.EM')]:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        print('  %-4s MISSING' % lab); continue
    if lab == '.sg':
        Lc = [l for l in open(p).read().splitlines() if l.strip()]
        h = next(i for i, l in enumerate(Lc) if len(l.split()) == 3 and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
        nn = int(Lc[h].split()[0]); xy = np.array([[float(Lc[h + 1 + k].split()[1]), float(Lc[h + 1 + k].split()[2])] for k in range(nn)])
    else:
        try:
            xy = np.loadtxt(p, skiprows=2)[:, :2]
        except Exception:
            xy = np.loadtxt(p)[:, :2]
    print('  %-4s x[%.3f, %.3f]  y[%.3f, %.3f]  (n=%d)' % (lab, xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max(), len(xy)))
