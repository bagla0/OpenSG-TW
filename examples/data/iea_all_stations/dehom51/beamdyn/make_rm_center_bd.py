'''make_rm_center_bd.py -- RM BeamDyn from the CENTER-REF 6x6 (build_rm_bundle ref='center'),
matching the center-ref 1D yaml. 51 stations, trapezoidal, 1500 Pa. Runs beamdyn + extracts FF.
Prefix iea51rmc.  Mass from the OpenSG_RM_Shell .out (faithful shell mass, static solve).'''
import os, sys, subprocess
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BD = os.path.join(ROOT, 'beamdyn_iea')
XSEC = os.path.abspath(os.path.join(ROOT, '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, BD); sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm
from beamdyn_trans import transformMatrixToBeamDyn, write_beamdyn_prop
from gen_beamdyn_51 import parse_out

NSTA = 51
ETAS = np.arange(NSTA) / 50.0
TAGS = ['iea_s%02d' % i for i in range(NSTA)]
BDDRV = '/home/roger/a/bagla0/miniconda3/bin/beamdyn_driver'
ENV = dict(os.environ); ENV['LD_LIBRARY_PATH'] = '/home/roger/a/bagla0/miniconda3/lib'

print('RM center-ref homogenization (build_rm_bundle ref=center) x 51 ...', flush=True)
K = np.zeros((NSTA, 6, 6)); M = np.zeros((NSTA, 6, 6))
for i in range(NSTA):
    K[i] = np.asarray(dehom_rm.build_rm_bundle(
        os.path.join(ROOT, 'shell51/1d_yaml/%s_shell.yaml' % TAGS[i]), ref='center')['Timo'])
    _, M[i] = parse_out(os.path.join(ROOT, 'shell51/out/OpenSG_RM_Shell/%s_OpenSG_RM_Shell.out' % TAGS[i]))
    if i % 10 == 0:
        print('  station %d' % i, flush=True)
np.savetxt(os.path.join(HERE, 'rm_center_K6x6_51.dat'), K.reshape(NSTA, 36), fmt='%.8e',
           header='RM build_rm_bundle CENTER-ref 6x6 (VABS order), 51 stations')

Kb, Mb = transformMatrixToBeamDyn(K.copy(), M.copy())
pf = write_beamdyn_prop(HERE, 'iea51rmc', ETAS, Kb, Mb, [1e-3] * 6)
os.replace(os.path.join(HERE, pf), os.path.join(HERE, 'iea51rmc_bd_props.inp'))
prim = open(os.path.join(BD, 'iea51_shell_bd_primary.inp')).read().replace(
    'iea51_shell_bd_props.inp', 'iea51rmc_bd_props.inp')
open(os.path.join(HERE, 'iea51rmc_bd_primary.inp'), 'w').write(prim)
drv = open(os.path.join(BD, 'iea51_shell_bd_driver.inp')).read().replace(
    'iea51_shell_bd_primary.inp', 'iea51rmc_bd_primary.inp')
open(os.path.join(HERE, 'iea51rmc_bd_driver.inp'), 'w').write(drv)
print('wrote iea51rmc props/primary/driver', flush=True)

r = subprocess.run([BDDRV, 'iea51rmc_bd_driver.inp'], cwd=HERE, env=ENV,
                   capture_output=True, text=True, timeout=600)
ok = os.path.exists(os.path.join(HERE, 'iea51rmc_bd_driver.out'))
print('ran iea51rmc : %s' % ('OK' if ok else 'FAIL\n' + r.stdout[-600:] + r.stderr[-600:]), flush=True)

if ok:
    L = [l for l in open(os.path.join(HERE, 'iea51rmc_bd_driver.out')).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith('Time'):
            h = l.split(); row = np.array([r.split() for r in L[i + 2:]], float)[-1]
            N = sum(1 for x in h if x.endswith('_TDxr') and x.startswith('N'))
            ff = []
            for k in range(1, N + 1):
                g = lambda n: row[h.index('N%03d_%s' % (k, n))]
                ff.append([g('FzL'), -g('FyL'), g('FxL'), g('MzL'), -g('MyL'), g('MxL')])
            ff = np.array(ff)
            np.savetxt(os.path.join(HERE, 'ff51_rmc_reform.dat'), np.column_stack([ETAS, ff]),
                       fmt='%.8e', header='eta F1 F2 F3 M1 M2 M3 (RM CENTER-ref BeamDyn, VABS order)')
            print('r0.2 RM-center FF = %s' % np.array2string(ff[10], precision=4))
            print('tip TipTDxr = %.4f m' % row[h.index('TipTDxr')])
            break
print('done -> dehom51/beamdyn/iea51rmc_bd_driver.out + ff51_rmc_reform.dat')
