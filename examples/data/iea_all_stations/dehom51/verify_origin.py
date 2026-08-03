'''prove the JAX/RM-vs-VABS.K gap is purely the LE-vs-(0,0) reference: parallel-axis shift VABS.K
(at LE) to the (0,0) reference axis and compare to homo_rm / homo_jax on ALL 6 diag terms.'''
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
DX = float(np.loadtxt(os.path.join(HERE, 'coords', 'frame_shift.dat')))   # LE-frame x of the ref axis


def vabs_timo(p):
    L = open(p).read().splitlines()
    for i, l in enumerate(L):
        if 'timoshenko stiffness' in l.lower():
            rows = []
            for l2 in L[i + 1:]:
                try:
                    v = [float(x) for x in l2.split()]
                    if len(v) >= 6:
                        rows.append(v[:6])
                except ValueError:
                    pass
                if len(rows) == 6:
                    break
            return np.array(rows)


def txt6(sub, pref, i):
    p = os.path.join(ROOT, 'shell51', sub, '%s_iea_s%02d.txt' % (pref, i))
    return np.loadtxt(p) if os.path.exists(p) else None


def Tshift(t2, t3):
    # strain(new ref at (t2,t3)) = T @ strain(old); order [e1,g12,g13,k1,k2,k3]
    T = np.eye(6)
    T[0, 4] = t3; T[0, 5] = -t2      # axial strain picks up k2*t3 - k3*t2
    T[1, 3] = -t3                    # gamma12 picks up -k1*t3
    T[2, 3] = t2                     # gamma13 picks up +k1*t2
    return T


# find DY (ref-axis y in LE frame): match VABS EI2 too; scan small t3, use t2=DX
i = 10
V = vabs_timo(os.path.join(ROOT, 'dehom_iea/sg_v201/iea_s%02d.sg.K' % i))
rm = txt6('homo_rm', 'OpenSG_RM', i)
jx = txt6('homo_jax', 'OpenSG_JAX', i)
print('s10  VABS.K(LE) diag = %s' % np.array2string(np.diag(V), precision=3))
print('     homo_rm(0,0) diag = %s' % np.array2string(np.diag(rm), precision=3))

best = None
for t2 in (DX, -DX):
    for t3 in np.linspace(-0.3, 0.3, 61):
        T = Tshift(t2, t3)
        Ks = np.linalg.inv(T).T @ V @ np.linalg.inv(T)     # VABS at LE -> shifted ref
        err = np.max(np.abs((np.diag(Ks) - np.diag(rm)) / np.diag(rm))) * 100
        if best is None or err < best[0]:
            best = (err, t2, t3, Ks)
err, t2, t3, Ks = best
print('\nBest shift: t2=%.3f  t3=%.3f  -> max diag %%err vs homo_rm = %.2f%%' % (t2, t3, err))
print('VABS.K shifted diag = %s' % np.array2string(np.diag(Ks), precision=3))
print('%-6s %s' % ('term', '  '.join('%8s' % t for t in LBL)))
print('%-6s %s' % ('vs rm', '  '.join('%+7.1f' % e for e in 100 * (np.diag(Ks) - np.diag(rm)) / np.diag(rm))))
if jx is not None:
    print('%-6s %s' % ('vs jax', '  '.join('%+7.1f' % e for e in 100 * (np.diag(Ks) - np.diag(jx)) / np.diag(jx))))

# apply the SAME (t2,t3) to all stations and report the diag match
print('\n=== VABS.K shifted to (0,0) vs homo_jax, all stations (max diag %err) ===')
T = Tshift(t2, t3)
worst = 0.0
for i in range(51):
    kf = os.path.join(ROOT, 'dehom_iea/sg_v201/iea_s%02d.sg.K' % i)
    jj = txt6('homo_jax', 'OpenSG_JAX', i)
    if not os.path.exists(kf) or jj is None:
        continue
    V = vabs_timo(kf)
    Ks = np.linalg.inv(T).T @ V @ np.linalg.inv(T)
    e = np.max(np.abs((np.diag(Ks) - np.diag(jj)) / np.diag(jj))) * 100
    worst = max(worst, e)
    if i in (0, 10, 25, 40, 50) or e > 10:
        print('  s%02d max diag %%err = %.1f%%' % (i, e))
print('WORST across stations = %.1f%%' % worst)
