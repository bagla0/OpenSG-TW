'''confirm the bug across all 51 stations: raw homo_rm vs homo_jax (should be <5%), and how far the
JAX_Solid.out has drifted from raw homo_jax. Plot the CORRECT raw-homo %err.'''
import os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def txt(sub, pref, i):
    p = os.path.join(ROOT, 'shell51', sub, '%s_iea_s%02d.txt' % (pref, i))
    return np.loadtxt(p) if os.path.exists(p) else None


def outmat(sub, i):
    p = os.path.join(ROOT, 'shell51/out', sub, 'iea_s%02d_%s.out' % (i, sub))
    if not os.path.exists(p):
        return None
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
            return np.array(rows)


eta = np.arange(51) / 50.0
Eraw = np.full((51, 6), np.nan)     # homo_rm vs homo_jax
Edrift = np.full((51, 6), np.nan)   # JAX_Solid.out vs raw homo_jax
for i in range(51):
    rm = txt('homo_rm', 'OpenSG_RM', i)
    jx = txt('homo_jax', 'OpenSG_JAX', i)
    jo = outmat('OpenSG_JAX_Solid', i)
    if rm is not None and jx is not None:
        Eraw[i] = 100 * (np.diag(rm) - np.diag(jx)) / np.diag(jx)
    if jx is not None and jo is not None:
        Edrift[i] = 100 * (np.diag(jo) - np.diag(jx)) / np.diag(jx)

print('=== RAW homo:  homo_rm vs homo_jax  (the CORRECT comparison) ===')
for k in range(6):
    e = np.abs(Eraw[:, k]); e = e[np.isfinite(e)]
    print('  %-4s mean %5.2f%%  max %6.2f%%' % (LBL[k], e.mean(), e.max()))
print('  overall max = %.2f%%  -> %s' % (np.nanmax(np.abs(Eraw)),
      'ALL <5%' if np.nanmax(np.abs(Eraw)) < 5 else 'some >5%'))

print('\n=== DRIFT: JAX_Solid.out vs raw homo_jax  (the corruption in the .out emission) ===')
for k in range(6):
    e = np.abs(Edrift[:, k]); e = e[np.isfinite(e)]
    print('  %-4s mean %5.2f%%  max %6.2f%%' % (LBL[k], e.mean(), e.max()))

# plot the CORRECT raw-homo %err
cols = plt.cm.rainbow(np.linspace(0, 1, 6))
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
for k, ax in enumerate(axes.ravel()):
    ax.axhline(0, color='0.6', lw=1, ls=':'); ax.axhspan(-5, 5, color='0.85', alpha=0.4, zorder=0)
    ax.plot(eta, Eraw[:, k], '-o', color=cols[k], ms=4, lw=1.6)
    ax.text(0.03, 0.94, LBL[k], transform=ax.transAxes, va='top', fontsize=11, weight='bold', color=cols[k])
    ax.set_ylabel('% error (RM vs JAX-solid)'); ax.grid(alpha=.25)
    if k >= 3:
        ax.set_xlabel('spanwise position  eta')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pcterr_homo_RAW_rm_vs_jax.png'), dpi=140, bbox_inches='tight')
print('\nwrote out/pcterr_homo_RAW_rm_vs_jax.png  (raw homo, the true <5% plot)')
