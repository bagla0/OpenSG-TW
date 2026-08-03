'''compare RM .out vs a FRESHLY re-emitted JAX .out (from current homo_jax) -> should now be <5%,
proving the old JAX_Solid.out was stale.'''
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def outdiag(sub, suffix, i):
    p = os.path.join(ROOT, 'shell51/out', sub, 'iea_s%02d_%s.out' % (i, suffix))
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
            return np.diag(np.array(rows))


E_old = np.full((51, 6), np.nan); E_new = np.full((51, 6), np.nan)
for i in range(51):
    rm = outdiag('OpenSG_RM_Shell', 'OpenSG_RM_Shell', i)
    jold = outdiag('OpenSG_JAX_Solid', 'OpenSG_JAX_Solid', i)
    jnew = outdiag('OpenSG_JAX_Solid_fresh', 'OpenSG_JAX_Solid_fresh', i)
    if rm is not None and jold is not None:
        E_old[i] = 100 * (rm - jold) / jold
    if rm is not None and jnew is not None:
        E_new[i] = 100 * (rm - jnew) / jnew

for name, E in [('RM.out vs OLD JAX_Solid.out (stale)', E_old),
                ('RM.out vs FRESH JAX .out (re-emitted)', E_new)]:
    print('=== %s ===' % name)
    for k in range(6):
        e = np.abs(E[:, k]); e = e[np.isfinite(e)]
        if len(e):
            print('  %-4s mean %5.2f%%  max %6.2f%%' % (LBL[k], e.mean(), e.max()))
    print('  overall max = %.2f%%\n' % np.nanmax(np.abs(E)))
