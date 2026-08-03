"""plots.py -- through-thickness figures mirroring Garg et al. (2023) figs 3, 4, 7.

Curves: exact 3-D elasticity (reference), FSDT (Garg's baseline), MSG-VAM (this work).
No titles on the figures -- captions live in the paper.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exact_cyl import ExactCyl
from materials import MATDB
import cyl_models as CM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

C_EX = 'k'
C_FS = '#d62728'
C_MG = '#1f77b4'


def solve(thick, angles, mats, S, npl_sg=6, n_per_layer=81):
    thick = np.asarray(thick, float)
    h = float(thick.sum())
    ex = ExactCyl(thick, angles, mats, MATDB, S * h)
    obj = CM.build(thick, angles, mats, MATDB, n_per_layer=npl_sg, elem_order=3)
    E6 = CM.plate_strains(obj['A6'], ex.p)
    zc, sig_e, _, _ = ex.profile(n_per_layer=n_per_layer)
    fs = CM.fsdt_profile(obj, E6, ex.p, n_per_layer=n_per_layer)
    mg = CM.msg_profile(obj, E6, ex.p, n_per_layer=n_per_layer)
    return dict(h=h, zc=zc / h, ex=sig_e, fs=fs, mg=mg, S=S, thick=thick)


def _panel(ax, r, comp, xlabel):
    z = r['zc']
    if comp == 's13':
        ax.plot(r['ex'][:, 4], z, C_EX, lw=2.4, label='3-D elasticity')
        ax.plot(r['fs']['s13'], z, C_FS, lw=1.6, ls='--', label='FSDT')
        ax.plot(r['mg']['s13'], z, C_MG, lw=1.6, ls='-.', label='MSG-VAM')
    else:
        ax.plot(r['ex'][:, 2], z, C_EX, lw=2.4, label='3-D elasticity')
        ax.plot(r['mg']['s33'], z, C_MG, lw=1.6, ls='-.', label='MSG-VAM')
    for zi in np.cumsum(r['thick'])[:-1] / r['h'] - 0.5:
        ax.axhline(zi, color='0.75', lw=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$z/h$')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(alpha=0.25, lw=0.5)


def fig_cross_ply():
    for S in (100, 10, 5):
        r = solve([1/3]*3, [0., 90., 0.], ['pagano']*3, S)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        _panel(axes[0], r, 's13', r'$\sigma_{13}/q_0$')
        _panel(axes[1], r, 's33', r'$\sigma_{33}/q_0$')
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'crossply_S{S}.png'), dpi=180,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote crossply_S{S}.png")


def fig_sandwich():
    for S in (20, 10, 5):
        r = solve([0.1, 0.8, 0.1], [0.]*3, ['face', 'core', 'face'], S, npl_sg=8)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        _panel(axes[0], r, 's13', r'$\sigma_{13}/q_0$')
        _panel(axes[1], r, 's33', r'$\sigma_{33}/q_0$')
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'sandwich_S{S}.png'), dpi=180,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote sandwich_S{S}.png")


def fig_convergence():
    Ss = np.array([4, 5, 6, 8, 10, 14, 20, 30, 50, 70, 100])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, (nm, thick, ang, mats, npl) in zip(axes, [
            ('cross-ply', [1/3]*3, [0., 90., 0.], ['pagano']*3, 6),
            ('sandwich', [0.1, 0.8, 0.1], [0.]*3, ['face', 'core', 'face'], 8)]):
        e13f, e13m, e33m, e11m = [], [], [], []
        for S in Ss:
            r = solve(thick, ang, mats, int(S), npl_sg=npl, n_per_layer=61)
            def rel(a, b):
                return 100 * np.linalg.norm(a - b) / np.linalg.norm(b)
            e13f.append(rel(r['fs']['s13'], r['ex'][:, 4]))
            e13m.append(rel(r['mg']['s13'], r['ex'][:, 4]))
            e33m.append(rel(r['mg']['s33'], r['ex'][:, 2]))
            e11m.append(rel(r['mg']['s11'], r['ex'][:, 0]))
        ax.loglog(Ss, e13f, color=C_FS, marker='s', ms=4, label=r'FSDT $\sigma_{13}$')
        ax.loglog(Ss, e13m, color=C_MG, marker='o', ms=4, label=r'MSG $\sigma_{13}$')
        ax.loglog(Ss, e33m, color='#2ca02c', marker='^', ms=4, label=r'MSG $\sigma_{33}$')
        ax.loglog(Ss, e11m, color='#9467bd', marker='v', ms=4, label=r'MSG $\sigma_{11}$')
        ax.set_xlabel(r'$S = L/h$   (' + nm + ')')
        ax.set_ylabel('relative $L_2$ error vs 3-D elasticity  [%]')
        ax.grid(which='both', alpha=0.25, lw=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'convergence.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  wrote convergence.png")


if __name__ == '__main__':
    fig_cross_ply()
    fig_sandwich()
    fig_convergence()
    print(f"figures -> {OUT}")
