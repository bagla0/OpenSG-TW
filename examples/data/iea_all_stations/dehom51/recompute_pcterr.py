'''recompute the HM %-error plot: RM Timo diag vs JAX-solid Timo diag, all 51 stations, 6 diag terms.
Two RM sources: (A) build_rm_bundle = the RM used by the DEHOM (paper ring), (B) OpenSG_RM_Shell =
the pipeline RM homogenizer (the one previously validated <5%).  JAX = OpenSG_JAX_Solid .K.'''
import os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
BD = os.path.join(HERE, 'beamdyn')
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
ETAS = np.arange(51) / 50.0
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
TAGS = ['iea_s%02d' % i for i in range(51)]


def out_diag(sub, suffix):
    D = np.full((51, 6), np.nan)
    for i in range(51):
        p = os.path.join(ROOT, 'shell51/out', sub, '%s_%s.out' % (TAGS[i], suffix))
        if not os.path.exists(p):
            continue
        L = open(p).read().splitlines()
        for j, l in enumerate(L):
            if l.strip().startswith('Stiffness'):
                rows = []; k = j + 1
                while len(rows) < 6 and k < len(L):
                    v = L[k].split()
                    try:
                        fv = [float(x) for x in v]
                        if len(fv) >= 6:
                            rows.append(fv[:6])
                    except ValueError:
                        pass
                    k += 1
                if len(rows) == 6:
                    D[i] = np.diag(np.array(rows))
                break
    return D


# JAX-solid diag (reference)
jx = out_diag('OpenSG_JAX_Solid', 'OpenSG_JAX_Solid')
# (A) dehom RM = build_rm_bundle (saved by reform_beamdyn)
rmA = np.loadtxt(os.path.join(BD, 'rm_K6x6_51.dat')).reshape(51, 6, 6)
rmA = np.array([np.diag(rmA[i]) for i in range(51)])
# (B) pipeline RM = OpenSG_RM_Shell
rmB = out_diag('OpenSG_RM_Shell', 'OpenSG_RM_Shell')

errA = 100.0 * (rmA - jx) / jx
errB = 100.0 * (rmB - jx) / jx

for name, err in [('build_rm_bundle (dehom RM)', errA), ('OpenSG_RM_Shell (pipeline RM)', errB)]:
    print('\n=== %s  vs  JAX-solid : |%%err| mean / max over 51 stations ===' % name)
    for k in range(6):
        e = np.abs(err[:, k]); e = e[np.isfinite(e)]
        print('  %-4s  mean %5.2f%%   max %6.2f%%  (at eta=%.2f)' %
              (LBL[k], e.mean(), e.max(), ETAS[np.nanargmax(np.abs(err[:, k]))]))
    allm = np.nanmax(np.abs(err))
    print('  overall max |%%err| = %.2f%%  -> %s' % (allm, '<5% all stations' if allm < 5 else 'exceeds 5%'))

# plot: 6 panels, both RM sources
fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for k in range(6):
    a = ax[k // 3, k % 3]
    a.plot(ETAS, errB[:, k], '-o', ms=3, color='#1f77b4', label='pipeline RM (OpenSG_RM_Shell)')
    a.plot(ETAS, errA[:, k], '--s', ms=3, color='#d62728', label='dehom RM (build_rm_bundle)')
    a.axhline(5, color='0.6', lw=.8, ls=':'); a.axhline(-5, color='0.6', lw=.8, ls=':')
    a.set_title('%s  %% error (RM vs JAX)' % LBL[k]); a.grid(alpha=.3)
    if k >= 3:
        a.set_xlabel('eta = r/R')
    if k == 0:
        a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pcterr_rm_vs_jax_diag.png'), dpi=150); plt.close(fig)
print('\nwrote out/pcterr_rm_vs_jax_diag.png')
# save the data
np.savetxt(os.path.join(OUT, 'pcterr_rm_vs_jax_diag.dat'),
           np.column_stack([ETAS, errB, errA]), fmt='%.4f',
           header='eta  %s(pipelineRM)  %s(dehomRM)' % ('/'.join(LBL), '/'.join(LBL)))
