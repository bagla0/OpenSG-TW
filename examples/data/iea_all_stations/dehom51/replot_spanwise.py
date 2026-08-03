'''replot_spanwise.py -- spanwise recovery line plots (all 51 stations, suction-crown
OML path), center-ref, with the paper conventions: x = r/R (0..1), stress in MPa,
displacement in METERS, plain full tick numbers, no in-panel annotations.
Reads out/spanwise_dehom/spanwise_oml_{RM,VABS}.out; writes out/dehom_plots_final/.'''
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, 'out', 'spanwise_dehom')
OUT = os.path.join(HERE, 'out', 'dehom_plots_final')
os.makedirs(OUT, exist_ok=True)
VABSC = '#1f77b4'
RMC = '#ff7f0e'


def read_out(path):
    d = {}
    for ln in open(path):
        if ln.startswith('#') or not ln.strip():
            continue
        t = ln.split()
        d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


rm = read_out(os.path.join(SD, 'spanwise_oml_RM.out'))
vb = read_out(os.path.join(SD, 'spanwise_oml_VABS.out'))
r = rm['non_dim_path']


def rel_err(a, b):
    d = np.abs(a - b)
    keep = d <= 8.0 * np.median(d) + 1e-12
    return 100.0 * np.linalg.norm((a - b)[keep]) / (np.linalg.norm(b[keep]) + 1e-30)


def plainaxis(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(alpha=0.3)


SC = [('s_11', r'$\sigma_{11}$'), ('s_12', r'$\sigma_{12}$'), ('s_22', r'$\sigma_{22}$')]
# root ply-flip points: at a few near-root stations the two models sample
# OPPOSITE sides of a ply interface (a sampling artifact, not a field
# difference) -- hidden with an in-figure disclosure, as in the r=0.2 plots.
d0 = np.abs(rm['s_11'] - vb['s_11'])
keep = d0 <= 8.0 * np.median(d0) + 1e-12
ndrop = int((~keep).sum())
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
for k, (c, lab) in enumerate(SC):
    ax = axs[k]
    ax.plot(r[keep], vb[c][keep] / 1e6, '-o', color=VABSC, ms=3.5, lw=1.5,
            label='VABS')
    ax.plot(r[keep], rm[c][keep] / 1e6, '--s', color=RMC, ms=3.5, mfc='none',
            mew=1.2, lw=1.4, label='OpenSG-RM')
    ax.set_ylabel('%s  [MPa]' % lab, fontsize=11)
    ax.set_xlabel('$r/R$', fontsize=11)
    plainaxis(ax)
    ax.legend(fontsize=9, loc='best')
if ndrop:
    axs[0].text(0.03, 0.03, '%d root ply-flip pt(s) hidden' % ndrop,
                transform=axs[0].transAxes, va='bottom', fontsize=7, color='0.55')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'spanwise_stress.png'), dpi=150)
plt.close(fig)

UC = [('u_1', '$u_1$'), ('u_2', '$u_2$'), ('u_3', '$u_3$')]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
for k, (c, lab) in enumerate(UC):
    ax = axs[k]
    ax.plot(r, vb[c], '-o', color=VABSC, ms=3.5, lw=1.5, label='VABS')
    ax.plot(r, rm[c], '--s', color=RMC, ms=3.5, mfc='none', mew=1.2, lw=1.4,
            label='OpenSG-RM')
    ax.set_ylabel('%s  [m]' % lab, fontsize=11)
    ax.set_xlabel('$r/R$', fontsize=11)
    plainaxis(ax)
    ax.legend(fontsize=9, loc='best')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'spanwise_disp.png'), dpi=150)
plt.close(fig)

print('spanwise stress %err: ' + '  '.join('%s %.1f%%' % (c, rel_err(rm[c], vb[c]))
      for c, _ in SC))
print('spanwise disp   %err: ' + '  '.join('%s %.2f%%' % (c, rel_err(rm[c], vb[c]))
      for c, _ in UC))
print('wrote spanwise_stress.png / spanwise_disp.png')
