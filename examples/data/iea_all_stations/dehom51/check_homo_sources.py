'''Which stored 6x6 == the RM dehom homo (build_rm_bundle) and which == JAX-solid .K?
Compare at r0.2 (s10) so I recompute only what is missing for the reformulated BeamDyn.'''
import os, sys, glob
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm


def read6(path, key='Stiffness'):
    L = open(path).read().splitlines()
    for i, l in enumerate(L):
        if key.lower() in l.lower():
            rows = []; j = i + 1
            while len(rows) < 6 and j < len(L):
                v = L[j].split()
                try:
                    fv = [float(x) for x in v]
                    if len(fv) >= 6:
                        rows.append(fv[:6])
                except ValueError:
                    pass
                j += 1
            if len(rows) == 6:
                return np.array(rows)
    return None


lbl = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
Kbun = np.asarray(dehom_rm.build_rm_bundle(os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml'), ref='oml')['Timo'])
print('build_rm_bundle (RM dehom) diag =', np.array2string(np.diag(Kbun), precision=3))

cands = {
    'homo_rm/OpenSG_RM_iea_s10.txt': os.path.join(ROOT, 'shell51/homo_rm/OpenSG_RM_iea_s10.txt'),
    'homo_jax/OpenSG_JAX_iea_s10.txt': os.path.join(ROOT, 'shell51/homo_jax/OpenSG_JAX_iea_s10.txt'),
    'out/OpenSG_RM_Shell/.out': os.path.join(ROOT, 'shell51/out/OpenSG_RM_Shell/iea_s10_OpenSG_RM_Shell.out'),
    'out/OpenSG_Hybrid_Solid/.out': os.path.join(ROOT, 'shell51/out/OpenSG_Hybrid_Solid/iea_s10_OpenSG_Hybrid_Solid.out'),
    'out/OpenSG_JAX_Solid/.out': os.path.join(ROOT, 'shell51/out/OpenSG_JAX_Solid/iea_s10_OpenSG_JAX_Solid.out'),
}
for nm, p in cands.items():
    if not os.path.exists(p):
        print('  %-32s MISSING' % nm); continue
    K = read6(p)
    if K is None:
        print('  %-32s no 6x6 parsed' % nm); continue
    fro = np.linalg.norm(K - Kbun) / np.linalg.norm(Kbun) * 100
    print('  %-32s diag=%s  Frob-vs-bundle=%.2f%%'
          % (nm, np.array2string(np.diag(K), precision=3), fro))

print('\nfile counts:')
for d in ('homo_rm', 'homo_jax', 'out/OpenSG_RM_Shell', 'out/OpenSG_Hybrid_Solid', 'out/OpenSG_JAX_Solid'):
    n = len(glob.glob(os.path.join(ROOT, 'shell51', d, '*ea_s*')))
    print('  %-28s %d files' % (d, n))
