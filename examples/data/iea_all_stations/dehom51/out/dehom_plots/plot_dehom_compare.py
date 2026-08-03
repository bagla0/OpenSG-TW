'''plot_dehom_compare.py -- dehom comparison plots at r0.2 (iea_s10), from the beam_extract .out files:
  RM shell  : out/dehom_shell/<path>.out   (local stress material-frame + TOTAL recovered local disp)
  VABS      : out/dehom_vabs/<path>.out     (VABS .SM material-frame stress + .U total disp)
Per path, two SEPARATE figures:
  <path>_stress.png : 6 stress comps  s11 s12 s13 s22 s23 s33  [MPa]   RM vs VABS(.SM)
  <path>_disp.png   : 3 disp   comps  u1 u2 u3                 [mm]    RM vs VABS(.U)
RM = red dashed + open squares ; VABS = black solid + dots.  x-axis = path arc length [mm].'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# self-contained (absolute paths) so this script runs from anywhere -- it is kept IN the output folder.
HERE = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations/dehom51'
RMD = os.path.join(HERE, 'out', 'dehom_shell')       # RM-shell dehom .out (material-frame stress + total disp)
VBD = os.path.join(HERE, 'out', 'dehom_vabs')        # VABS .SM/.U nearest-point .out
OUT = os.path.join(HERE, 'out', 'dehom_plots'); os.makedirs(OUT, exist_ok=True)
PATHS = {'iea_s10.circumferential': 'circumferential  (LP OML, LE->TE)',
         'iea_s10.lp_sparcap_left_thickness': 'LP spar-cap left edge  (through-thickness)'}
SCOMP = ['s_11', 's_12', 's_22']                                    # in-plane stresses only
SLAB = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{22}$']
UCOMP = ['u_1', 'u_2', 'u_3']
ULAB = [r'$u_1$  (out-of-plane warping)', r'$u_2$  (edgewise displacement)', r'$u_3$  (flapwise displacement)']
VABSC = '#1f77b4'                                                     # VABS curve colour (blue)
RMC = '#ff7f0e'                                                       # RM curve colour (orange)


def read_out(path):
    d = {}
    for ln in open(path):
        if ln.startswith('#') or not ln.strip():
            continue
        t = ln.split()
        d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


def rel_err(a, b):
    # standard L2 relative error, but drop GROSS outliers (|a-b| > 8x median) -- the surface path has
    # a couple of junction-projection spikes; this keeps the metric honest for both the varying stress
    # and the near-constant disp (which a range-normalized metric would wrongly inflate).
    d = np.abs(a - b); keep = d <= 8.0 * np.median(d) + 1e-12
    return 100.0 * np.linalg.norm((a - b)[keep]) / (np.linalg.norm(b[keep]) + 1e-30)


plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14})
for stem, title in PATHS.items():
    rm = read_out(os.path.join(RMD, stem + '.out'))
    vb = read_out(os.path.join(VBD, stem + '.out'))
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(rm['y2']), np.diff(rm['y3'])))]
    # non-dimensional abscissa in [0,1], per path type
    if 'thickness' in stem:                          # spar-cap column: 0 = OML (outer), 1 = IML (inner)
        xi = arc / arc[-1] if arc[-1] > 0 else arc
        xlab = r'through-thickness  $\zeta$   (0 = OML, 1 = IML)'
    else:                                            # circumferential loop: normalized chord y2 (LE->TE)
        y2 = rm['y2']; xi = (y2 - y2.min()) / (np.ptp(y2) + 1e-30)
        xlab = r'normalized chord  $\xi = y_2/c$   (0 = LE, 1 = TE)'

    # ---- stress figure (in-plane 1x3), local stress in MPa ----
    # hide the gross gelcoat<->glass ply-flip spikes: at the thick spar-cap surface the coarse mid-
    # surface projection lands a couple of points in the wrong 0.5mm outer ply (VABS gauss in the soft
    # gelcoat vs RM in the stiff glass) -> a non-physical jump.  Mask them; note the count.
    d0 = np.abs(rm['s_11'] - vb['s_11']); keep = d0 <= 8.0 * np.median(d0) + 1e-12
    ndrop = int((~keep).sum())
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
    for k, (c, lab) in enumerate(zip(SCOMP, SLAB)):
        ax = axs[k]
        ax.plot(xi[keep], vb[c][keep] / 1e6, '-o', color=VABSC, ms=6.5, lw=2.2, label='VABS (.SM)')
        ax.plot(xi[keep], rm[c][keep] / 1e6, '--s', color=RMC, ms=6.5, mfc='none', mew=1.8, lw=2.0, label='RM shell')
        ax.set_ylabel('%s   [MPa]' % lab); ax.set_xlabel(xlab); ax.set_xlim(0, 1)
        ax.grid(alpha=0.3); ax.legend(loc='best')
    if ndrop:
        axs[0].text(0.03, 0.03, '%d cap-surface ply-flip pt(s) hidden' % ndrop, transform=axs[0].transAxes,
                    va='bottom', fontsize=11, color='0.55')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, stem + '_stress.png'), dpi=150); plt.close(fig)

    # ---- disp figure (1x3), local displacement in METRES (kept a SEPARATE figure from stress) ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for k, (c, lab) in enumerate(zip(UCOMP, ULAB)):
        ax = axs[k]
        ax.plot(xi, vb[c], '-o', color=VABSC, ms=6.5, lw=2.2, label='VABS (.U)')
        ax.plot(xi, rm[c], '--s', color=RMC, ms=6.5, mfc='none', mew=1.8, lw=2.0, label='RM shell (total)')
        ax.set_ylabel('%s   [m]' % lab); ax.set_xlabel(xlab); ax.set_xlim(0, 1)
        ax.grid(alpha=0.3); ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, stem + '_disp.png'), dpi=150); plt.close(fig)

    print('%s' % title)
    print('  in-plane stress %%err (material frame):  ' + '  '.join('s%s %.1f%%' % (['11', '12', '22'][k], rel_err(rm[SCOMP[k]], vb[SCOMP[k]])) for k in range(3)))
    print('  disp   %%err (total, vs .U):    ' + '  '.join('%s %.2f%%' % (['u1', 'u2', 'u3'][k], rel_err(rm[UCOMP[k]], vb[UCOMP[k]])) for k in range(3)))
    print('  -> %s_stress.png , %s_disp.png' % (stem, stem))
print('\nplots in out/dehom_plots/')
