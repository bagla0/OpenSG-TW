"""plots_navier.py -- replicate the figure set of Mendonca & Ruviaro (FEAD 260, 2026)
with the OpenSG-RM curve added.  One figure per (laminate, quantity); no titles.

Curves:  black  = analytic 3-D elasticity (Pagano-1970 configuration, state-space form)
         red    = analytic FSDT + the Mendonca-Ruviaro recovery chain (their process --
                  the curve their FE + sequential smoothing converges to)
         blue   = OpenSG-RM (MSG structure gene, direct recovery, no corrections)
         grey   = FSDT kinematic profile u0 + z psi_x (u panels only)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

import navier_models as NM   # noqa: E402

OUT = os.path.join(_HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

C3D, CFS, CMG, CKIN = 'k', '#d62728', '#1f77b4', '0.55'
ASPECTS = (4, 100)


def _interfaces(ax, r):
    thick, _, _ = NM.CASES[r['case']](r['H'])
    for zi in np.cumsum(thick)[:-1] / r['H'] - 0.5:
        ax.axhline(zi, color='0.8', lw=0.6, zorder=0)


def _panel(ax, r, comp):
    zb = r['zbar']
    n = r['nrm']
    if comp == 'txz':
        ax.plot(r['exact']['txz'] * n['t'], zb, C3D, lw=2.2, label='Analytic 3-D')
        ax.plot(r['fsdt']['txz'] * n['t'], zb, CFS, lw=1.6, ls='--',
                label='FSDT + M-R recovery')
        ax.plot(r['msg']['txz'] * n['t'], zb, CMG, lw=1.6, ls='-.', label='OpenSG-RM')
        ax.set_xlabel(r'$\bar\tau_{xz}$')
    elif comp == 'sz':
        ax.plot(r['exact']['sz'] * n['s'], zb, C3D, lw=2.2, label='Analytic 3-D')
        ax.plot(r['fsdt']['sz'] * n['s'], zb, CFS, lw=1.6, ls='--',
                label='FSDT + M-R recovery')
        ax.plot(r['msg']['sz'] * n['s'], zb, CMG, lw=1.6, ls='-.', label='OpenSG-RM')
        ax.set_xlabel(r'$\bar\sigma_{z}$')
    elif comp == 'u':
        d = r['fsdt']['d']
        ax.plot(r['exact']['u'] * n['u'], zb, C3D, lw=2.2, label='Analytic 3-D')
        ax.plot(r['fsdt']['u'] * n['u'], zb, CFS, lw=1.6, ls='--',
                label='FSDT + M-R recovery')
        ax.plot(r['msg']['u'] * n['u'], zb, CMG, lw=1.6, ls='-.', label='OpenSG-RM')
        ax.plot((d[0] + r['zc'] * d[3]) * n['u'], zb, CKIN, lw=1.2, ls=':',
                label='FSDT kinematic')
        ax.set_xlabel(r'$\bar u$')
    elif comp == 'w':
        ax.plot(r['exact']['w'] * n['w'], zb, C3D, lw=2.2, label='Analytic 3-D')
        ax.plot(r['fsdt']['w'] * n['w'], zb, CFS, lw=1.6, ls='--',
                label='FSDT + M-R recovery')
        ax.plot(r['msg']['w'] * n['w'], zb, CMG, lw=1.6, ls='-.', label='OpenSG-RM')
        ax.set_xlabel(r'$\bar w$')
    _interfaces(ax, r)
    ax.set_ylabel(r'$\bar z$')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(alpha=0.25, lw=0.5)


def fig_quantity(case, comp, aspects=ASPECTS):
    rs = [NM.run_case(case, a_, n_out=61) for a_ in aspects]
    fig, axes = plt.subplots(1, len(rs), figsize=(4.5 * len(rs), 3.6), squeeze=False)
    for ax, r in zip(axes[0], rs):
        _panel(ax, r, comp)
        ax.text(0.04, 0.03, f"$a/H={r['aspect']}$", transform=ax.transAxes,
                fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    name = f"{case}_{comp}.png"
    fig.savefig(os.path.join(OUT, name), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name}")
    return rs


def error_table():
    rows = []
    print(f"\n{'case':<9} {'a/H':>4} {'qty':<4} {'M-R rec':>10} {'OpenSG-RM':>10}")
    for case in ('sym', 'asym', 'sandwich'):
        for aspect in ASPECTS:
            r = NM.run_case(case, aspect, n_out=61)
            for comp in ('txz', 'sz', 'u', 'w'):
                efs = NM.relerr(r['fsdt'][comp], r['exact'][comp])
                ems = NM.relerr(r['msg'][comp], r['exact'][comp])
                rows.append((case, aspect, comp, efs, ems))
                print(f"{case:<9} {aspect:>4} {comp:<4} {100*efs:>9.3f}% "
                      f"{100*ems:>9.3f}%")
    import csv
    path = os.path.join(_HERE, 'results')
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'table_navier.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['case', 'aspect', 'quantity', 'fsdt_mr', 'opensg_rm'])
        w.writerows(rows)
    print(f"wrote {os.path.join(path, 'table_navier.csv')}")


if __name__ == '__main__':
    for case in ('sym', 'asym', 'sandwich'):
        for comp in ('txz', 'sz', 'w'):
            fig_quantity(case, comp)
        fig_quantity(case, 'u', aspects=(4,))
    error_table()
    print(f"figures -> {OUT}")
