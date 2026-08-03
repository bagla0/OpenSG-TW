'''plot_dehom_center_final.py -- FINAL r=0.2 path comparison figures, center-ref
(the stored dehom51 pipeline), with the paper plotting conventions:
  * x-axis: non-dimensional path coordinate 0..1 (from the .out non_dim_path row)
  * displacements in METERS, plain full tick numbers (no offset/multiplier)
  * no in-panel percent annotations (numbers go to the console for the prose)
Reads out/dehom_shell (RM center) + out/dehom_vabs; writes out/dehom_plots_final/.'''
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RMD = os.path.join(HERE, 'out', 'dehom_shell')
VBD = os.path.join(HERE, 'out', 'dehom_vabs')
OUT = os.path.join(HERE, 'out', 'dehom_plots_final')
os.makedirs(OUT, exist_ok=True)
PATHS = {'iea_s10.circumferential': 'circumferential',
         'iea_s10.lp_sparcap_left_thickness': 'cap thickness'}
SCOMP = ['s_11', 's_12', 's_22']
SLAB = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{22}$']
UCOMP = ['u_1', 'u_2', 'u_3']
ULAB = [r'$u_1$', r'$u_2$', r'$u_3$']
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


def rel_err(a, b):
    d = np.abs(a - b)
    keep = d <= 8.0 * np.median(d) + 1e-12
    return 100.0 * np.linalg.norm((a - b)[keep]) / (np.linalg.norm(b[keep]) + 1e-30)


def plainaxis(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(alpha=0.3)


for stem, title in PATHS.items():
    rm = read_out(os.path.join(RMD, stem + '.out'))
    vb = read_out(os.path.join(VBD, stem + '.out'))
    s = rm['non_dim_path']
    s = (s - s.min()) / (s.max() - s.min() + 1e-30)          # 0..1

    d0 = np.abs(rm['s_11'] - vb['s_11'])
    keep = d0 <= 8.0 * np.median(d0) + 1e-12
    ndrop = int((~keep).sum())

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    for k, (c, lab) in enumerate(zip(SCOMP, SLAB)):
        ax = axs[k]
        ax.plot(s[keep], vb[c][keep] / 1e6, '-o', color=VABSC, ms=3.5, lw=1.5,
                label='VABS')
        ax.plot(s[keep], rm[c][keep] / 1e6, '--s', color=RMC, ms=3.5, mfc='none',
                mew=1.2, lw=1.4, label='OpenSG-RM')
        ax.set_ylabel('%s  [MPa]' % lab, fontsize=11)
        ax.set_xlabel('non-dimensional path coordinate', fontsize=10)
        plainaxis(ax)
        ax.legend(fontsize=9, loc='best')
    if ndrop:
        axs[0].text(0.03, 0.03, '%d ply-interface pt(s) hidden' % ndrop,
                    transform=axs[0].transAxes, va='bottom', fontsize=7, color='0.55')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, stem + '_stress.png'), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, (c, lab) in enumerate(zip(UCOMP, ULAB)):
        ax = axs[k]
        ax.plot(s, vb[c], '-o', color=VABSC, ms=3.5, lw=1.5, label='VABS')
        ax.plot(s, rm[c], '--s', color=RMC, ms=3.5, mfc='none', mew=1.2, lw=1.4,
                label='OpenSG-RM')
        ax.set_ylabel('%s  [m]' % lab, fontsize=11)
        ax.set_xlabel('non-dimensional path coordinate', fontsize=10)
        plainaxis(ax)
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, stem + '_disp.png'), dpi=150)
    plt.close(fig)

    print('%s (center-ref)' % title)
    print('  stress: ' + '  '.join('s%s %.1f%%' % (['11', '12', '22'][k],
          rel_err(rm[SCOMP[k]], vb[SCOMP[k]])) for k in range(3)))
    print('  disp  : ' + '  '.join('%s %.2f%%' % (['u1', 'u2', 'u3'][k],
          rel_err(rm[UCOMP[k]], vb[UCOMP[k]])) for k in range(3)))
print('figures ->', OUT)
