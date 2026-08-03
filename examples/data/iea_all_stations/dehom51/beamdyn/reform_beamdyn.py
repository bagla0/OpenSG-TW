'''reform_beamdyn.py -- two dehom-CONSISTENT BeamDyn models + FF one-to-one check.
  RM   BeamDyn : K = build_rm_bundle (the exact RM dehom homo), M from OpenSG_RM_Shell .out
  solid BeamDyn: K,M from OpenSG_JAX_Solid .out (the JAX-solid .K homo, drives the .glb/VABS dehom)
Both use the same trapezoidal 51-station reference line + 1500 Pa load (copied from the existing
iea51_shell/solid primary+driver).  Runs beamdyn_driver, extracts per-station FF (VABS order,
section-local -> B swap), and tabulates RM vs solid FF at every station.
'''
import os, sys, shutil, subprocess
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))                  # dehom51/beamdyn
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))            # iea_all_stations
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


def SHELL(i): return os.path.join(ROOT, 'shell51/1d_yaml/%s_shell.yaml' % TAGS[i])
def JAXO(i):  return os.path.join(ROOT, 'shell51/out/OpenSG_JAX_Solid/%s_OpenSG_JAX_Solid.out' % TAGS[i])
def RMO(i):   return os.path.join(ROOT, 'shell51/out/OpenSG_RM_Shell/%s_OpenSG_RM_Shell.out' % TAGS[i])


# 1. gather K,M ---------------------------------------------------------------
print('computing RM (build_rm_bundle) 6x6 for 51 stations ...', flush=True)
Krm = np.zeros((NSTA, 6, 6)); Mrm = np.zeros((NSTA, 6, 6))
Kjx = np.zeros((NSTA, 6, 6)); Mjx = np.zeros((NSTA, 6, 6))
for i in range(NSTA):
    Krm[i] = np.asarray(dehom_rm.build_rm_bundle(SHELL(i), ref='oml')['Timo'])
    _, Mrm[i] = parse_out(RMO(i))
    Kjx[i], Mjx[i] = parse_out(JAXO(i))
    if i % 10 == 0:
        print('  station %d done' % i, flush=True)

np.savetxt(os.path.join(HERE, 'rm_K6x6_51.dat'), Krm.reshape(NSTA, 36), fmt='%.8e',
           header='RM build_rm_bundle 6x6 (VABS order), row-major, 51 stations')
np.savetxt(os.path.join(HERE, 'jax_K6x6_51.dat'), Kjx.reshape(NSTA, 36), fmt='%.8e',
           header='JAX-solid .K 6x6 (VABS order), row-major, 51 stations')

# 2. transform + write props/primary/driver ----------------------------------
Krmb, Mrmb = transformMatrixToBeamDyn(Krm.copy(), Mrm.copy())
Kjxb, Mjxb = transformMatrixToBeamDyn(Kjx.copy(), Mjx.copy())
for pref, K, M, srcpref in [('iea51rm', Krmb, Mrmb, 'iea51_shell'),
                            ('iea51jax', Kjxb, Mjxb, 'iea51_solid')]:
    pf = write_beamdyn_prop(HERE, pref, ETAS, K, M, [1e-3] * 6)
    os.replace(os.path.join(HERE, pf), os.path.join(HERE, '%s_bd_props.inp' % pref))
    prim = open(os.path.join(BD, '%s_bd_primary.inp' % srcpref)).read().replace(
        '%s_bd_props.inp' % srcpref, '%s_bd_props.inp' % pref)
    open(os.path.join(HERE, '%s_bd_primary.inp' % pref), 'w').write(prim)
    drv = open(os.path.join(BD, '%s_bd_driver.inp' % srcpref)).read().replace(
        '%s_bd_primary.inp' % srcpref, '%s_bd_primary.inp' % pref)
    open(os.path.join(HERE, '%s_bd_driver.inp' % pref), 'w').write(drv)
print('wrote props/primary/driver for iea51rm + iea51jax', flush=True)

# 3. run beamdyn --------------------------------------------------------------
for pref in ('iea51rm', 'iea51jax'):
    r = subprocess.run([BDDRV, '%s_bd_driver.inp' % pref], cwd=HERE, env=ENV,
                       capture_output=True, text=True, timeout=600)
    ok = os.path.exists(os.path.join(HERE, '%s_bd_driver.out' % pref))
    print('ran %s : %s' % (pref, 'OK' if ok else 'FAIL\n' + r.stdout[-800:] + r.stderr[-800:]), flush=True)


# 4. extract FF + compare -----------------------------------------------------
def ff51(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith('Time'):
            h = l.split(); row = np.array([r.split() for r in L[i + 2:]], float)[-1]
            N = sum(1 for x in h if x.endswith('_TDxr') and x.startswith('N'))
            out = []
            for k in range(1, N + 1):
                g = lambda n: row[h.index('N%03d_%s' % (k, n))]
                Fx, Fy, Fz, Mx, My, Mz = g('FxL'), g('FyL'), g('FzL'), g('MxL'), g('MyL'), g('MzL')
                out.append([Fz, -Fy, Fx, Mz, -My, Mx])           # VABS order
            return np.array(out)


FFrm = ff51(os.path.join(HERE, 'iea51rm_bd_driver.out'))
FFjx = ff51(os.path.join(HERE, 'iea51jax_bd_driver.out'))
np.savetxt(os.path.join(HERE, 'ff51_rm_reform.dat'),
           np.column_stack([ETAS, FFrm]), fmt='%.8e', header='eta F1 F2 F3 M1 M2 M3 (RM BeamDyn, VABS order)')
np.savetxt(os.path.join(HERE, 'ff51_jax_reform.dat'),
           np.column_stack([ETAS, FFjx]), fmt='%.8e', header='eta F1 F2 F3 M1 M2 M3 (JAX-solid BeamDyn, VABS order)')

lbl = ['F1', 'F2', 'F3', 'M1', 'M2', 'M3']
print('\n=== FF one-to-one: RM BeamDyn vs JAX-solid BeamDyn (max |%%diff| over dominant comps) ===')
print('%3s %6s  %12s %12s %12s  %8s' % ('st', 'eta', 'F3(shear)', 'M2(flapbend)', 'M1(torsion)', 'max%diff'))
worst = 0.0
for i in range(NSTA):
    a, b = FFrm[i], FFjx[i]
    dom = np.abs(a) > 1e-2 * np.max(np.abs(a))
    pd = np.max(np.abs((a[dom] - b[dom]) / a[dom])) * 100 if dom.any() else 0.0
    worst = max(worst, pd)
    if i % 5 == 0 or pd > 2:
        print('%3d %6.3f  %12.4e %12.4e %12.4e  %7.2f%%' % (i, ETAS[i], a[2], a[4], a[3], pd))
print('WORST dominant-component FF diff across all 51 stations = %.2f%%' % worst)
print('  (FF is load/equilibrium-driven, so one-to-one is expected regardless of the 6x6)')
