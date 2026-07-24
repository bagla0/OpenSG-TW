"""plots.py -- every figure in the paper.  No titles: the caption is the title.

  crossply_S*.png   through-thickness sigma13 / sigma33, three-layer cross-ply
  sandwich_S*.png   ditto, sandwich (the case where FSDT collapses)
  convergence.png   relative L2 error vs slenderness S, log-log
  sweep.png         error distribution over the whole Garg-2023 sampling box
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jaxcfg import jnp          # noqa: F401  (x64 first)
import models as M
from materials import MATDB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
RES = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)

C_EX, C_FS, C_MG = 'k', '#d62728', '#1f77b4'
CASES = {
    'crossply': ([1 / 3] * 3, [0., 90., 0.], ['pagano'] * 3, 6),
    'sandwich': ([0.1, 0.8, 0.1], [0.] * 3, ['face', 'core', 'face'], 8),
}


def _panel(ax, r, comp, xlabel):
    z = r['zc'] / r['h']
    if comp == 's13':
        ax.plot(r['exact'][:, 4], z, C_EX, lw=2.4, label='3-D elasticity')
        ax.plot(r['fsdt']['s13'], z, C_FS, lw=1.6, ls='--', label='FSDT')
        ax.plot(r['msg']['s13'], z, C_MG, lw=1.6, ls='-.', label='MSG-VAM')
    else:
        ax.plot(r['exact'][:, 2], z, C_EX, lw=2.4, label='3-D elasticity')
        ax.plot(r['msg']['s33'], z, C_MG, lw=1.6, ls='-.', label='MSG-VAM')
    for zi in np.cumsum(r['thick'])[:-1] / r['h'] - 0.5:
        ax.axhline(zi, color='0.75', lw=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$z/h$')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(alpha=0.25, lw=0.5)


def fig_profiles(tag, slendernesses):
    thick, ang, mats, npl = CASES[tag]
    for S in slendernesses:
        r = M.run(thick, ang, mats, MATDB, S, n_per_layer_out=81, npl_sg=npl)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        _panel(axes[0], r, 's13', r'$\sigma_{13}/q_0$')
        _panel(axes[1], r, 's33', r'$\sigma_{33}/q_0$')
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'{tag}_S{S}.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"  {tag}_S{S}.png")


def fig_convergence():
    Ss = [4, 5, 6, 8, 10, 14, 20, 30, 50, 70, 100]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, tag in zip(axes, ('crossply', 'sandwich')):
        thick, ang, mats, npl = CASES[tag]
        cur = {k: [] for k in ('f13', 'm13', 'm33', 'm11')}
        for S in Ss:
            r = M.run(thick, ang, mats, MATDB, S, n_per_layer_out=61, npl_sg=npl)
            ex = r['exact']
            cur['f13'].append(100 * M.relerr(r['fsdt']['s13'], ex[:, 4]))
            cur['m13'].append(100 * M.relerr(r['msg']['s13'], ex[:, 4]))
            cur['m33'].append(100 * M.relerr(r['msg']['s33'], ex[:, 2]))
            cur['m11'].append(100 * M.relerr(r['msg']['s11'], ex[:, 0]))
        ax.loglog(Ss, cur['f13'], color=C_FS, marker='s', ms=4,
                  label=r'FSDT $\sigma_{13}$')
        ax.loglog(Ss, cur['m13'], color=C_MG, marker='o', ms=4,
                  label=r'MSG $\sigma_{13}$')
        ax.loglog(Ss, cur['m33'], color='#2ca02c', marker='^', ms=4,
                  label=r'MSG $\sigma_{33}$')
        ax.loglog(Ss, cur['m11'], color='#9467bd', marker='v', ms=4,
                  label=r'MSG $\sigma_{11}$')
        ax.set_xlabel(r'$S = L/h$   (' + tag + ')')
        ax.set_ylabel(r'relative $L_2$ error vs 3-D elasticity  [\%]'.replace('\\%', '%'))
        ax.grid(which='both', alpha=0.25, lw=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'convergence.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  convergence.png")


def fig_sweep():
    path = os.path.join(RES, 'sweep.npz')
    if not os.path.exists(path):
        print("  (skipping sweep.png -- run sweep.py first)")
        return
    d = np.load(path, allow_pickle=True)
    err = 100 * d['err']
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    series = [('FSDT $\\sigma_{13}$', err[:, 0], C_FS),
              ('MSG $\\sigma_{13}$', err[:, 1], C_MG),
              ('MSG $\\sigma_{33}$', err[:, 2], '#2ca02c'),
              ('MSG $\\sigma_{11}$', err[:, 3], '#9467bd')]
    for nm, e, c in series:
        xs = np.sort(e)
        axes[0].plot(xs, np.linspace(0, 100, xs.size), color=c, lw=1.8, label=nm)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'relative $L_2$ error vs 3-D elasticity  [%]')
    axes[0].set_ylabel(r'cumulative share of laminates  [%]')
    axes[0].grid(which='both', alpha=0.25, lw=0.5)

    S = d['S']
    axes[1].semilogy(S, err[:, 0], '.', ms=3, color=C_FS, alpha=0.5,
                     label='FSDT $\\sigma_{13}$')
    axes[1].semilogy(S, err[:, 1], '.', ms=3, color=C_MG, alpha=0.6,
                     label='MSG $\\sigma_{13}$')
    axes[1].semilogy(S, err[:, 2], '.', ms=3, color='#2ca02c', alpha=0.6,
                     label='MSG $\\sigma_{33}$')
    axes[1].set_xlabel(r'$S = L/h$')
    axes[1].set_ylabel(r'relative $L_2$ error  [%]')
    axes[1].grid(which='both', alpha=0.25, lw=0.5)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sweep.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  sweep.png")


if __name__ == '__main__':
    fig_profiles('crossply', (100, 10, 5))
    fig_profiles('sandwich', (20, 10, 5))
    fig_convergence()
    fig_sweep()
    print(f"figures -> {OUT}")
