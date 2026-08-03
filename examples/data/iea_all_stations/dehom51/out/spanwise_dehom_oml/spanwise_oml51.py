'''spanwise_oml51.py -- 51-station OML-referenced chain: homogenization %err vs VABS .K
AND the spanwise suction-crown dehomogenization, from the regenerated windIO-native OML
yamls (shell51/1d_yaml_oml51, reference recorded in each yaml).

Per station: build_rm_bundle (ref=oml from the yaml, MSG G) -> Timo 6x6 diag vs the
VABS .sg.K Timoshenko block; then the OML-top gauss-point stress + total-disp recovery
exactly as out/spanwise_dehom/spanwise_dehom.py (same landmark, same FF + BeamDyn
kinematics -- the internal loads are model-insensitive).

Outputs (this folder): rm_vs_vabs_diag_oml.dat, dehom_homo_pcterr_51_oml.png,
spanwise .coords + RM/VABS .out, spanwise_oml_stress.png, spanwise_oml_disp.png.'''
import os
import sys

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DEHOM = os.path.abspath(os.path.join(HERE, '..', '..'))
ROOT = os.path.abspath(os.path.join(DEHOM, '..'))
VABS = os.path.join(DEHOM, 'out', 'VABS_iea51')
SHELLD = os.path.join(ROOT, 'shell51', '1d_yaml_oml51')
XSEC = os.path.abspath(os.path.join(ROOT, '..', '..', 'TW-paper', 'xsec_paper'))
sys.path.insert(0, XSEC)
import jax

jax.config.update('jax_enable_x64', True)
import dehom_rm

FF_ALL = np.loadtxt(os.path.join(DEHOM, 'beamdyn', 'ff51_rmc_reform.dat'))
BD_OUT = os.path.join(DEHOM, 'beamdyn', 'iea51rmc_bd_driver.out')
BE = ('11', '12', '13', '22', '23', '33')
SVOIGT = {'11': 0, '12': 5, '13': 4, '22': 1, '23': 3, '33': 2}


def beam_kinematics(path, node):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith('Time'):
            h = l.split(); row = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: row[h.index('N%03d_%s' % (node, nm))]
            TD = np.array([g('TDxr'), g('TDyr'), g('TDzr')]); RD = np.array([g('RDxr'), g('RDyr'), g('RDzr')])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]]); return u_g, C
    raise ValueError('no BeamDyn header')


def block6(L, key):
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
            return np.array(rows)
    return None


eta, y2, y3, RMS, RMU, VBS, VBU = [], [], [], [], [], [], []
diag_rows, fails = [], []
for i in range(51):
    smp = os.path.join(VABS, 'iea_s%02d.sg.SM' % i)
    up = os.path.join(VABS, 'iea_s%02d.sg.U' % i)
    kp = os.path.join(VABS, 'iea_s%02d.sg.K' % i)
    shp = os.path.join(SHELLD, 'iea_s%02d_shell.yaml' % i)
    if not all(os.path.exists(p) for p in (smp, up, kp, shp)):
        fails.append((i, 'missing file')); continue
    try:
        B = dehom_rm.build_rm_bundle(shp)               # ref read from the yaml -> oml
        C6 = np.asarray(B['Timo'])
        Kv = block6([l for l in open(kp).read().splitlines()], 'Timoshenko Stiffness Matrix')
        de = 100.0 * (np.diag(C6) - np.diag(Kv)) / np.diag(Kv)
        diag_rows.append([i / 50.0] + list(de))

        SM = np.loadtxt(smp, skiprows=2); U = np.loadtxt(up)
        band = 0.10
        sel = np.abs(SM[:, 0]) < band
        if sel.sum() < 3:
            sel = np.abs(SM[:, 0]) < 0.30
        cand = np.where(sel)[0]
        itop = int(cand[np.argmax(SM[cand, 1])]); pt = SM[itop, :2]
        Vs = SM[itop, 2:8]
        uxy = U[:, 1:3]; uv = U[:, 3:6]
        dU, iU = cKDTree(uxy).query(pt[None], k=4)
        wv = 1.0 / (dU + 1e-8 * (dU.sum(1, keepdims=True) + 1e-30)); wv /= wv.sum(1, keepdims=True)
        Vu = np.einsum('pk,pkj->pj', wv, uv[iU])[0]
        FF = FF_ALL[i, 1:]
        S = np.asarray(dehom_rm.stress_at_points(B, pt[None], beam_force_vabs=FF,
                                                 frame='material', n_per_layer=4,
                                                 flow_avg=True)['stress'])[0]
        W = np.asarray(dehom_rm.disp_at_points(B, pt[None], beam_force_vabs=FF))[0]
        u_g, C = beam_kinematics(BD_OUT, i + 1); r3 = np.array([0.0, pt[0], pt[1]])
        Ut = u_g + C @ (W + r3) - r3
        eta.append(i / 50.0); y2.append(pt[0]); y3.append(pt[1])
        RMS.append([S[SVOIGT[k]] for k in BE]); RMU.append(Ut)
        VBS.append(Vs.tolist()); VBU.append(Vu.tolist())
        print('s%02d ok  GA3err %+6.2f%%  s11 RM %.1f / V %.1f MPa   u3 RM %.3f / V %.3f m'
              % (i, de[2], S[0] / 1e6, Vs[0] / 1e6, Ut[2], Vu[2]), flush=True)
    except Exception as e:
        fails.append((i, str(e)[:70])); print('s%02d FAIL: %s' % (i, str(e)[:70]), flush=True)

eta = np.array(eta); y2 = np.array(y2); y3 = np.array(y3)
RMS = np.array(RMS); RMU = np.array(RMU); VBS = np.array(VBS); VBU = np.array(VBU)
diag_rows = np.array(diag_rows)
print('\n%d/51 stations ok ; fails: %s' % (len(eta), fails))
np.savetxt(os.path.join(HERE, 'rm_vs_vabs_diag_oml.dat'), diag_rows,
           header='eta  %err diag EA GA2 GA3 GJ EI2 EI3  (RM shell OML MSG-G vs VABS .K)')

# ---- homogenization %err panels (same style as dehom_homo_pcterr_51) ----
LBL = ['EA', 'GA_2', 'GA_3', 'GJ', 'EI_2', 'EI_3']
TIT = ['extension $EA$', 'transv. shear $GA_2$', 'transv. shear $GA_3$',
       'torsion $GJ$', 'flap bending $EI_2$', 'edge bending $EI_3$']
col = plt.cm.rainbow(np.linspace(0, 1, 6))
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 17, 'xtick.labelsize': 14,
                     'ytick.labelsize': 14, 'legend.fontsize': 14})
fig, axs = plt.subplots(3, 2, figsize=(12, 13.5))
for k in range(6):
    ax = axs.flat[k]
    ax.axhspan(-5, 5, color='0.9', zorder=0); ax.axhline(0, color='0.6', lw=1.2, ls=':')
    ax.plot(diag_rows[:, 0], diag_rows[:, 1 + k], '-o', color=col[k], mec='k', mew=0.5, ms=8, lw=2.2)
    mx = np.nanmax(np.abs(diag_rows[:, 1 + k]))
    ax.set_ylim(-max(6, 1.2 * mx), max(6, 1.2 * mx))
    ax.set_xlabel(r'span $r/R$'); ax.set_ylabel(r'$%s$  RM vs VABS  [\%%]' % LBL[k])
    ax.set_title(TIT[k], fontsize=15)
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'dehom_homo_pcterr_51_oml.png'), dpi=150)
plt.close(fig)
for k in range(6):
    print('  %-4s mean %5.2f%%  max %6.2f%%' % (LBL[k], np.nanmean(np.abs(diag_rows[:, 1 + k])),
                                                np.nanmax(np.abs(diag_rows[:, 1 + k]))))

np.savetxt(os.path.join(HERE, 'iea.spanwise_oml.coords'), np.column_stack([eta, y2, y3]),
           fmt='%.9e', header='eta(r/R)  y2  y3   OML-top gauss point per station, (0,0) frame')


def write_out(path, hdr, S6, U3):
    with open(path, 'w') as f:
        f.write('# %s ; abscissa = r/R ; beam_extract row-keyed layout\n' % hdr)
        f.write('non_dim_path ' + ' '.join('%.6e' % e for e in eta) + '\n')
        f.write('y2 ' + ' '.join('%.6e' % v for v in y2) + '\n')
        f.write('y3 ' + ' '.join('%.6e' % v for v in y3) + '\n')
        for j, k in enumerate(BE):
            f.write('s_%s ' % k + ' '.join('%.6e' % v for v in S6[:, j]) + '\n')
        for j, k in enumerate(('1', '2', '3')):
            f.write('u_%s ' % k + ' '.join('%.6e' % v for v in U3[:, j]) + '\n')


write_out(os.path.join(HERE, 'spanwise_oml_RM.out'),
          'RM-shell (OML ref, MSG G) spanwise: stress[Pa,material] + TOTAL disp[m]', RMS, RMU)
write_out(os.path.join(HERE, 'spanwise_oml_VABS.out'),
          'VABS spanwise: .SM stress[Pa,material] + .U disp[m]', VBS, VBU)

VABSC = '#1f77b4'; RMC = '#ff7f0e'


def relerr(a, b):
    d = np.abs(a - b); keep = d <= 8.0 * np.median(d) + 1e-12
    return 100.0 * np.linalg.norm((a - b)[keep]) / (np.linalg.norm(b[keep]) + 1e-30)


SIN = [('11', 0), ('12', 1), ('22', 3)]
SLAB = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{22}$']
d0 = np.abs(RMS[:, 0] - VBS[:, 0]); keep = d0 <= 8.0 * np.median(d0) + 1e-12
ndrop = int((~keep).sum())
fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
for ax, (k, idx), lab in zip(axs, SIN, SLAB):
    ax.plot(eta[keep], VBS[keep, idx] / 1e6, '-o', color=VABSC, ms=6.5, lw=2.2, label='VABS (.SM)')
    ax.plot(eta[keep], RMS[keep, idx] / 1e6, '--s', color=RMC, ms=6.5, mfc='none', mew=1.8,
            lw=2.0, label='RM shell')
    ax.set_xlabel(r'span  $r/R$'); ax.set_ylabel('%s   [MPa]' % lab)
    ax.grid(alpha=0.3); ax.legend(loc='best')
if ndrop:
    axs[0].text(0.03, 0.03, '%d root ply-flip pt(s) hidden' % ndrop, transform=axs[0].transAxes,
                va='bottom', fontsize=11, color='0.55')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'spanwise_oml_stress.png'), dpi=150); plt.close(fig)

ULAB = [r'$u_1$ (out-of-plane warping)', r'$u_2$ (edgewise displacement)', r'$u_3$ (flapwise displacement)']
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for k, (ax, lab) in enumerate(zip(axs, ULAB)):
    ax.plot(eta, VBU[:, k], '-o', color=VABSC, ms=6.5, lw=2.2, label='VABS (.U)')
    ax.plot(eta, RMU[:, k], '--s', color=RMC, ms=6.5, mfc='none', mew=1.8, lw=2.0,
            label='RM shell (total)')
    ax.set_xlabel(r'span  $r/R$'); ax.set_ylabel('%s   [m]' % lab)
    ax.grid(alpha=0.3); ax.legend(loc='best')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'spanwise_oml_disp.png'), dpi=150); plt.close(fig)
print('stress relerr: ' + '  '.join('s%s %.1f%%' % (k, relerr(RMS[:, i], VBS[:, i]))
                                    for k, i in SIN))
print('disp relerr  : ' + '  '.join('u%d %.2f%%' % (k + 1, relerr(RMU[:, k], VBU[:, k]))
                                    for k in range(3)))
print('wrote dat + pcterr fig + coords + .out + 2 spanwise figs ->', HERE)
