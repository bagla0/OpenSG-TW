'''Two separate spanwise plots (style of Camarena Fig.1: nondim span x-axis, kN / kNm, circle markers):
  out/plots_rm_vs_vabs/flap_force_rm_vs_vabs.png : Flap force F3 [kN]  vs  r/R   (RM shell vs VABS)
  out/plots_rm_vs_vabs/torsion_rm_vs_vabs.png    : Torsional moment M1 [kNm] vs r/R  (RM shell vs VABS)
F3 = FxL (flap shear), M1 = MzL (torsion) from the BeamDyn .out; RM = iea51rmc (center-ref), VABS = iea51vabs.'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
BDR = os.path.join(HERE, 'beamdyn', 'iea51rmc_bd_driver.out')            # RM shell (center-ref)
BDV = os.path.join(HERE, 'out', 'VABS_iea51', 'iea51vabs_bd_driver.out')  # VABS .K
OUT = os.path.join(HERE, 'out', 'plots_rm_vs_vabs')
os.makedirs(OUT, exist_ok=True)


def ff(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith('Time'):
            h = l.split(); row = np.array([r.split() for r in L[i + 2:]], float)[-1]
            N = sum(1 for x in h if x.endswith('_TDxr') and x.startswith('N'))
            eta = np.array([(k - 1) / 50.0 for k in range(1, N + 1)])
            F3 = np.array([row[h.index('N%03d_FxL' % k)] for k in range(1, N + 1)])   # flap shear
            M1 = np.array([row[h.index('N%03d_MzL' % k)] for k in range(1, N + 1)])   # torsion
            return eta, F3, M1
    raise ValueError('no header ' + path)


er, F3r, M1r = ff(BDR)
ev, F3v, M1v = ff(BDV)

# ---- flap force ----
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(ev, F3v / 1e3, '-o', color='#1f77b4', ms=5, lw=1.6, label='VABS $.\\mathrm{K}$ (2-D solid)')
ax.plot(er, F3r / 1e3, '--s', color='#d62728', ms=5, mfc='none', mew=1.4, lw=1.6, label='RM shell')
ax.set_xlabel(r'Blade nondimensional span, $r/R$  [-]', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Flap force, $F_3$  [kN]', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1); ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=11, frameon=True)                  # curves are low at right -> clear
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'flap_force_rm_vs_vabs.png'), dpi=160); plt.close(fig)

# ---- torsion ----
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(ev, M1v / 1e3, '-o', color='#1f77b4', ms=5, lw=1.6, label='VABS $.\\mathrm{K}$ (2-D solid)')
ax.plot(er, M1r / 1e3, '--s', color='#d62728', ms=5, mfc='none', mew=1.4, lw=1.6, label='RM shell')
ax.set_xlabel(r'Blade nondimensional span, $r/R$  [-]', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Torsional moment, $M_1$  [kN$\cdot$m]', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1); ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=11, frameon=True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'torsion_rm_vs_vabs.png'), dpi=160); plt.close(fig)

# ---- combined dual-axis (Camarena Fig.1 layout: flap on left/blue, torsion on right/orange) ----
FLAP = '#1f77b4'   # blue   -> left  axis (flap force)
TORS = '#ff7f0e'   # orange -> right axis (torsion)
fig, axL = plt.subplots(figsize=(7.4, 4.7))
axR = axL.twinx()
l1, = axL.plot(ev, F3v / 1e3, '-o', color=FLAP, ms=5, lw=1.7, label=r'Flap force $F_3$ — VABS $.\mathrm{K}$')
l2, = axL.plot(er, F3r / 1e3, '--s', color=FLAP, ms=5, mfc='none', mew=1.4, lw=1.4, label=r'Flap force $F_3$ — RM shell')
l3, = axR.plot(ev, M1v / 1e3, '-o', color=TORS, ms=5, lw=1.7, label=r'Torsion $M_1$ — VABS $.\mathrm{K}$')
l4, = axR.plot(er, M1r / 1e3, '--s', color=TORS, ms=5, mfc='none', mew=1.4, lw=1.4, label=r'Torsion $M_1$ — RM shell')
axL.set_xlabel(r'Blade nondimensional span, $r/R$  [-]', fontsize=12, fontweight='bold')
axL.set_ylabel(r'Flap force, $F_3$  [kN]', fontsize=12, fontweight='bold', color=FLAP)
axR.set_ylabel(r'Torsional moment, $M_1$  [kN$\cdot$m]', fontsize=12, fontweight='bold', color=TORS)
axL.tick_params(axis='y', colors=FLAP); axR.tick_params(axis='y', colors=TORS)
axL.set_xlim(0, 1); axL.set_ylim(0, None); axR.set_ylim(0, None)
axL.grid(alpha=0.3)
lines = [l1, l2, l3, l4]
axL.legend(lines, [x.get_label() for x in lines], loc='upper right', fontsize=9.5, frameon=True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'flap_torsion_combined_rm_vs_vabs.png'), dpi=160); plt.close(fig)

print('wrote flap_force_rm_vs_vabs.png + torsion_rm_vs_vabs.png + flap_torsion_combined_rm_vs_vabs.png -> %s' % OUT)
print('  root  F3: RM %.1f kN  VABS %.1f kN' % (F3r[0] / 1e3, F3v[0] / 1e3))
print('  root  M1: RM %.1f kNm VABS %.1f kNm' % (M1r[0] / 1e3, M1v[0] / 1e3))
