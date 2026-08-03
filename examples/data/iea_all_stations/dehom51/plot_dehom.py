'''plot_dehom.py -- r0.2 dehom deliverables:
  out/r020_section_paths.png            : 2-D solid (by material) + the 2 dehom paths
  out/<path>_stress_<comp>.png/.dat     : RM shell vs VABS .SM, one file per stress component
  out/<path>_disp_<u>.png/.dat          : RM shell vs VABS .U,  one file per disp component
Also prints RM-vs-VABS errors in BOTH stress frames (material vs global) to settle the convention.'''
import os, sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
XSEC = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax; jax.config.update('jax_enable_x64', True)
import dehom_rm
from scipy.spatial import cKDTree

OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
COORDS = os.path.join(HERE, 'coords')
SGV = os.path.join(ROOT, 'dehom_iea/sg_v201')
SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')
DX = float(np.loadtxt(os.path.join(COORDS, 'frame_shift.dat')))
FF = np.loadtxt(os.path.join(HERE, 'beamdyn', 'ff51_rm_reform.dat'))[10, 1:]   # dehom-consistent RM FF
SCOMP = ['S11', 'S22', 'S33', 'S23', 'S13', 'S12']
UCOMP = ['u1', 'u2', 'u3']
PATHS = {'circumferential': 'iea_s10.circumferential',
         'lp_sparcap_left': 'iea_s10.lp_sparcap_left_thickness'}


def parse_sg(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    h = next(i for i, l in enumerate(L) if len(l.split()) == 3 and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
    nn, ne, nm = [int(x) for x in L[h].split()]
    xy = np.array([[float(L[h + 1 + k].split()[1]), float(L[h + 1 + k].split()[2])] for k in range(nn)])
    conn = [[int(x) for x in L[h + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
    egrp = np.array([int(L[h + 1 + nn + ne + k].split()[1]) for k in range(ne)])
    gmat = {int(L[h + 1 + nn + 2 * ne + k].split()[0]): int(L[h + 1 + nn + 2 * ne + k].split()[1]) for k in range(nm)}
    return xy, conn, np.array([gmat[g] for g in egrp])


# ---- section + paths PNG ----
xy, conn, emat = parse_sg(os.path.join(SGV, 'iea_s10.sg'))
polys = [xy[[n - 1 for n in c]] for c in conn]
cmap = plt.cm.tab10
fig, ax = plt.subplots(figsize=(13, 5))
ax.add_collection(PolyCollection(polys, facecolors=[cmap((emat[k] - 1) % 10) for k in range(len(conn))], edgecolors='none'))
for key, stem in PATHS.items():
    c = np.loadtxt(os.path.join(COORDS, stem + '.coords'))[:, :2]
    ax.plot(c[:, 0], c[:, 1], '-o', ms=2.5, lw=1.8, label=key)
ax.set_aspect('equal'); ax.autoscale(); ax.axis('off'); ax.legend(loc='upper right', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'r020_section_paths.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
print('wrote out/r020_section_paths.png  (LE-frame, carbon cap = material 4)')

# ---- dehom + per-component plots ----
B = dehom_rm.build_rm_bundle(SHELL, ref='center')          # center-ref = default (matches center-ref 1D yaml)
for pkey, stem in PATHS.items():
    c = np.loadtxt(os.path.join(COORDS, stem + '.coords'))[:, :2]
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))] * 1e3   # mm
    cref = c.copy(); cref[:, 0] -= DX
    Rg = dehom_rm.stress_at_points(B, cref, beam_force_vabs=FF, frame='global', n_per_layer=4)
    Rm = dehom_rm.stress_at_points(B, cref, beam_force_vabs=FF, frame='material', n_per_layer=4)
    Sg = np.asarray(Rg['stress']); Sm = np.asarray(Rm['stress'])
    U = np.asarray(dehom_rm.disp_at_points(B, cref, beam_force_vabs=FF))
    # VABS .SM (stress) + .U (disp) at these exact gauss points
    sm = np.loadtxt(os.path.join(SGV, 'iea_s10.sg.SM'), skiprows=2)
    smxy = sm[:, :2]; V = sm[:, 2:8][:, [0, 3, 5, 4, 2, 1]]
    _, si = cKDTree(smxy).query(c); Vs = V[si]
    u = np.loadtxt(os.path.join(SGV, 'iea_s10.sg.U'))
    uxy = u[:, 1:3]; Uv = u[:, 3:6]
    _, ui = cKDTree(uxy).query(c); Vu = Uv[ui]
    eg = np.linalg.norm(Sg - Vs) / np.linalg.norm(Vs) * 100
    em = np.linalg.norm(Sm - Vs) / np.linalg.norm(Vs) * 100
    print('%-16s stress ||RM-VABS||: global=%.1f%%  material=%.1f%%  -> use %s' %
          (pkey, eg, em, 'GLOBAL' if eg < em else 'material'))
    S = Sg if eg <= em else Sm
    for j, cmp in enumerate(SCOMP):
        np.savetxt(os.path.join(OUT, '%s_stress_%s.dat' % (pkey, cmp)),
                   np.column_stack([arc, S[:, j] / 1e6, Vs[:, j] / 1e6]), fmt='%.6e',
                   header='arc[mm]  RM_%s[MPa]  VABS_%s[MPa]' % (cmp, cmp))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(arc, Vs[:, j] / 1e6, 'k-o', ms=3, label='VABS (.SM)')
        ax.plot(arc, S[:, j] / 1e6, 'r--s', ms=3, label='RM shell')
        ax.set_xlabel('path arc length [mm]'); ax.set_ylabel('%s [MPa]' % cmp)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(OUT, '%s_stress_%s.png' % (pkey, cmp)), dpi=140); plt.close(fig)
    for j, cmp in enumerate(UCOMP):
        np.savetxt(os.path.join(OUT, '%s_disp_%s.dat' % (pkey, cmp)),
                   np.column_stack([arc, U[:, j] * 1e3, Vu[:, j] * 1e3]), fmt='%.6e',
                   header='arc[mm]  RM_%s[mm]  VABS_%s[mm]' % (cmp, cmp))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(arc, Vu[:, j] * 1e3, 'k-o', ms=3, label='VABS (.U)')
        ax.plot(arc, U[:, j] * 1e3, 'r--s', ms=3, label='RM shell')
        ax.set_xlabel('path arc length [mm]'); ax.set_ylabel('%s [mm]' % cmp)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(OUT, '%s_disp_%s.png' % (pkey, cmp)), dpi=140); plt.close(fig)
    print('  wrote 6 stress + 3 disp component files for %s' % pkey)
print('\nall deliverables in dehom51/out/')
