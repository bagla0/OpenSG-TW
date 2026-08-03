'''12 spanwise plots: RM-shell .out vs VABS .K Timoshenko diagonal (both at (0,0) reference axis).
  out/plots_rm_vs_vabs/stiff_<term>.png : the two stiffness curves (VABS star, RM square)
  out/plots_rm_vs_vabs/err_<term>.png   : % error of RM vs VABS (diamond markers, colored)
No figure titles (per convention). RM = shell51/out/OpenSG_RM_Shell ; VABS = dehom51/out/VABS_iea51.'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
VK = os.path.join(HERE, 'out', 'VABS_iea51')
OUT = os.path.join(HERE, 'out', 'plots_rm_vs_vabs')
os.makedirs(OUT, exist_ok=True)
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
TITLE = ['EA  (C11, extension)', 'GA2  (C22, transverse shear)', 'GA3  (C33, transverse shear)',
         'GJ  (C44, torsion)', 'EI2  (C55, flap bending)', 'EI3  (C66, edge bending)']
UNIT = ['GPa', 'GPa', 'GPa', 'GPa', 'GPa', 'GPa']          # stiffness scaled by 1e9
COL = plt.cm.tab10(np.linspace(0, 1, 6))


def diag_from(path, key):
    L = open(path).read().splitlines()
    for i, l in enumerate(L):
        if key.lower() in l.lower():
            rows = []; j = i + 1
            while len(rows) < 6 and j < len(L):
                try:
                    v = [float(x) for x in L[j].split()]
                    if len(v) >= 6:
                        rows.append(v[:6])
                except ValueError:
                    pass
                j += 1
            if len(rows) == 6:
                return np.diag(np.array(rows))
    return np.full(6, np.nan)


eta = np.arange(51) / 50.0
RM = np.array([diag_from(os.path.join(ROOT, 'shell51/out/OpenSG_RM_Shell/iea_s%02d_OpenSG_RM_Shell.out' % i),
                         'Stiffness') for i in range(51)])
VB = np.array([diag_from(os.path.join(VK, 'iea_s%02d.sg.K' % i), 'Timoshenko Stiffness Matrix')
               for i in range(51)])
err = 100.0 * (RM - VB) / VB

for k in range(6):
    # ---- stiffness value plot ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.semilogy(eta, VB[:, k] / 1e9, ls='-', color='crimson', marker='*', ms=11, mec='k', mew=0.4,
                label='VABS $.\\mathrm{K}$ (2-D solid)')
    ax.semilogy(eta, RM[:, k] / 1e9, ls='--', color='navy', marker='s', ms=6, mfc='none', mew=1.4,
                label='RM shell')
    ax.set_xlabel(r'spanwise position  $\eta = r/R$', fontsize=11)
    ax.set_ylabel(r'%s  [%s]' % (LBL[k], UNIT[k]), fontsize=11)
    ax.text(0.03, 0.06, TITLE[k], transform=ax.transAxes, fontsize=11, weight='bold', color='0.25')
    ax.grid(which='both', alpha=0.25); ax.legend(fontsize=10, frameon=True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'stiff_%s.png' % LBL[k]), dpi=150); plt.close(fig)
    # ---- % error plot ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhspan(-5, 5, color='0.85', alpha=0.5, zorder=0, label=r'$\pm5\%$ band')
    ax.axhline(0, color='0.5', lw=1, ls=':')
    ax.plot(eta, err[:, k], ls='-', color=COL[k], marker='D', ms=7, mec='k', mew=0.4)
    mx = np.nanmax(np.abs(err[:, k]))
    ax.set_ylim(-max(6, 1.15 * mx), max(6, 1.15 * mx))
    ax.set_xlabel(r'spanwise position  $\eta = r/R$', fontsize=11)
    ax.set_ylabel(r'%s  RM vs VABS  [\%% error]' % LBL[k], fontsize=11)
    ax.text(0.03, 0.92, '%s   (mean %.1f%%, max %.1f%%)' % (TITLE[k], np.nanmean(np.abs(err[:, k])), mx),
            transform=ax.transAxes, va='top', fontsize=10, weight='bold', color=COL[k])
    ax.grid(alpha=0.25); ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'err_%s.png' % LBL[k]), dpi=150); plt.close(fig)

np.savetxt(os.path.join(OUT, 'rm_vs_vabs_diag.dat'), np.column_stack([eta, RM, VB, err]), fmt='%.6e',
           header='eta | RM(EA GA2 GA3 GJ EI2 EI3) | VABS(...) | %err(...)')
print('wrote 12 plots -> out/plots_rm_vs_vabs/')
for k in range(6):
    print('  %-4s RM vs VABS: mean %5.2f%%  max %6.2f%%' % (LBL[k], np.nanmean(np.abs(err[:, k])), np.nanmax(np.abs(err[:, k]))))
