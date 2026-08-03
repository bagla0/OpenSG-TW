'''plot_pcterr.py -- spanwise %-error of the RM-shell Timoshenko 6x6 DIAGONAL vs the 2-D solid
(VABS-equivalent, validated <1% vs a real VABS .K), for the 49 stations that have BOTH, all at the
x1 reference axis. 6 subplots (EA, GA2, GA3, GJ, EI2, EI3). x = spanwise eta (0..1). No title;
labels/annotation only.'''
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RM = os.path.join(HERE, 'homo_rm')
JX = os.path.join(HERE, 'homo_jax')
LBL = ['EA  (C11)', 'GA2  (C22)', 'GA3  (C33)', 'GJ  (C44)', 'EI2  (C55)', 'EI3  (C66)']


def read6(path):
    return np.loadtxt(path)


def collect(d, pref):
    out = {}
    for f in glob.glob(os.path.join(d, pref + '*.txt')):
        tag = os.path.basename(f).replace(pref, '').replace('.txt', '')   # iea_sNN
        try:
            out[tag] = read6(f)
        except Exception:
            pass
    return out


sh = collect(RM, 'OpenSG_RM_')
so = collect(JX, 'OpenSG_JAX_')
common = sorted(set(sh) & set(so), key=lambda t: int(t.split('s')[-1]))
eta = np.array([int(t.split('s')[-1]) / 50.0 for t in common])
D = np.array([[100.0 * (sh[t][k, k] - so[t][k, k]) / so[t][k, k] for k in range(6)] for t in common])
print('stations with both shell+solid: %d  (eta %.2f..%.2f)' % (len(common), eta.min(), eta.max()))

cols = plt.cm.rainbow(np.linspace(0, 1, 6))
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
for k, ax in enumerate(axes.ravel()):
    ax.axhline(0, color='0.6', lw=1, ls=':')
    ax.axhspan(-3, 3, color='0.85', alpha=0.35, zorder=0)
    ax.plot(eta, D[:, k], '-o', color=cols[k], ms=4, lw=1.6)
    ax.set_ylabel('% error vs 2-D solid')
    ax.text(0.03, 0.94, LBL[k], transform=ax.transAxes, va='top', fontsize=11, weight='bold', color=cols[k])
    mx = np.nanmax(np.abs(D[:, k]))
    ax.set_ylim(-max(4, 1.15 * mx), max(4, 1.15 * mx))
    if k >= 3:
        ax.set_xlabel('spanwise position  η  (0 → 1)')
    ax.grid(alpha=0.25)
fig.tight_layout()
out = os.path.join(HERE, 'pcterr_shell_vs_solid.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print('wrote', out)
# text summary
for k in range(6):
    print('%-10s  mean|%%err|=%.2f  max|%%err|=%.2f' % (LBL[k], np.mean(np.abs(D[:, k])), np.max(np.abs(D[:, k]))))
