'''% error plot of the Timoshenko 6x6 DIAGONAL, RM .out vs JAX-solid .out, all 51 stations
(mid reference, origin (0,0)).  RM = out/OpenSG_RM_Shell ; JAX = out/OpenSG_JAX_Solid.'''
import os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
LBL = ['EA  (C11)', 'GA2  (C22)', 'GA3  (C33)', 'GJ  (C44)', 'EI2  (C55)', 'EI3  (C66)']


def diag(sub):
    D = np.full((51, 6), np.nan)
    for i in range(51):
        p = os.path.join(ROOT, 'shell51/out', sub, 'iea_s%02d_%s.out' % (i, sub))
        if not os.path.exists(p):
            continue
        L = open(p).read().splitlines()
        for j, l in enumerate(L):
            if l.strip().startswith('Stiffness'):
                rows = []; k = j + 1
                while len(rows) < 6 and k < len(L):
                    try:
                        fv = [float(x) for x in L[k].split()]
                        if len(fv) >= 6:
                            rows.append(fv[:6])
                    except ValueError:
                        pass
                    k += 1
                D[i] = np.diag(np.array(rows)); break
    return D


rm = diag('OpenSG_RM_Shell')
jx = diag('OpenSG_JAX_Solid')
eta = np.arange(51) / 50.0
ok = np.isfinite(rm[:, 0]) & np.isfinite(jx[:, 0])
E = 100.0 * (rm - jx) / jx

cols = plt.cm.rainbow(np.linspace(0, 1, 6))
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
for k, ax in enumerate(axes.ravel()):
    ax.axhline(0, color='0.6', lw=1, ls=':')
    ax.axhspan(-5, 5, color='0.85', alpha=0.4, zorder=0)
    ax.plot(eta[ok], E[ok, k], '-o', color=cols[k], ms=4, lw=1.6)
    ax.text(0.03, 0.94, LBL[k], transform=ax.transAxes, va='top', fontsize=11, weight='bold', color=cols[k])
    ax.set_ylabel('% error (RM vs JAX-solid)'); ax.grid(alpha=0.25)
    if k >= 3:
        ax.set_xlabel('spanwise position  eta  (0 -> 1)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'pcterr_homo_rm_vs_jax.png'), dpi=140, bbox_inches='tight')
np.savetxt(os.path.join(OUT, 'pcterr_homo_rm_vs_jax.dat'), np.column_stack([eta, E]),
           fmt='%.3f', header='eta  ' + '  '.join(['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']))
print('wrote out/pcterr_homo_rm_vs_jax.png  (%d stations)' % ok.sum())
for k in range(6):
    e = np.abs(E[ok, k])
    print('  %-9s mean %5.2f%%  max %6.2f%%' % (LBL[k], e.mean(), e.max()))
