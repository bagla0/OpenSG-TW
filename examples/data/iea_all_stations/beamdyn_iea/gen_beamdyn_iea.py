'''gen_beamdyn_iea.py  --  build the 3 BeamDyn input files for the IEA-22 blade from the
OpenSG-FEniCSx 2-D solid cross-section 6x6 (stiffness + mass), for a STATIC solve under the
1200 Pa flapwise surface traction.

Data source (FEniCSx solid Timoshenko 6x6 K and 6x6 M, VABS convention):
    ../out/OpenSG_FEniCSx_Solid/iea_rXXXX_OpenSG_FEniCSx_Solid.out   (16 stations)
    (identical to homo_fenics/OpenSG_FEniCSx_*.txt / OpenSG_Mass_*.txt, which are the labeled
     .dat form of the same data; the homo_fenics/ .txt files are not present so we parse the .out)

Writes into beamdyn_iea/:
    bd_props_iea.inp     (props : station_total=16, damp mu=1e-3, VABS->BeamDyn axis remap)
    bd_primary_iea.inp   (primary : 50 KP reference line, order_elem=10, quadrature=1 Gaussian)
    bd_driver_iea.inp    (driver : static, gravity=0, 51 uniform-eta flapwise point loads)

    ~/miniconda3/envs/opensg_2_0/bin/python gen_beamdyn_iea.py
'''
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))          # .../beamdyn_iea
ROOT = os.path.dirname(HERE)                                # .../iea_all_stations
BARURC = os.path.join(ROOT, 'beamdyn_barurc')
OUTDIR = os.path.join(ROOT, 'out', 'OpenSG_FEniCSx_Solid')
WINDIO = os.path.join(ROOT, 'IEA-22-280-RWT.yaml')
FFTXT = os.path.join(ROOT, 'ff_beam_load.txt')

sys.path.insert(0, HERE)
from beamdyn_trans import transformMatrixToBeamDyn, write_beamdyn_prop

P_PA = 1500.0     # flapwise surface traction [Pa]  -- single knob; scales the whole driver load table
NPL = 51          # driver point loads on a uniform eta grid 0,0.02,...,1.0 (matches BAR-URC)
# NKP (number of collinear key points) is set = round(L/2) once L is known (see PRIMARY section):
#   RULE  kp_total = int(round(blade_length / 2))  -> ~1 key point per 2 m of span.

# ------------------------------------------------------------------ station etas (windIO airfoil
# spanwise_position, sorted); these map 1:1 to the sorted station tags r%04d
ETAS = np.array([0.0, 0.02, 0.04868313468735217, 0.06649308419703373,
                 0.08345591801468194, 0.10220904193570582, 0.11036272014164386,
                 0.1364246061019374, 0.15564440587515185, 0.19665336575444797,
                 0.24696148735364706, 0.3992636115637571, 0.5335887750152993,
                 0.738938689884722, 0.9799991709122947, 1.0])
TAGS = ['r%04d' % round(e * 1000) for e in ETAS]


def parse_out(path):
    """Read the 6x6 Stiffness and 6x6 Mass blocks from a labeled OpenSG .out file."""
    lines = open(path).read().splitlines()

    def block(key):
        for i, l in enumerate(lines):
            if l.strip().startswith(key):
                return np.array([[float(x) for x in lines[j].split()] for j in range(i + 1, i + 7)])
        raise ValueError('no %r block in %s' % (key, path))

    return block('Stiffness'), block('Mass')


def load_KM():
    K = np.zeros((16, 6, 6))
    M = np.zeros((16, 6, 6))
    for i, tag in enumerate(TAGS):
        p = os.path.join(OUTDIR, 'iea_%s_OpenSG_FEniCSx_Solid.out' % tag)
        K[i], M[i] = parse_out(p)
    return K, M


def windio_axes():
    d = yaml.safe_load(open(WINDIO))
    ra = d['components']['blade']['reference_axis']
    gx, vx = np.array(ra['x']['grid']), np.array(ra['x']['values'])
    gy, vy = np.array(ra['y']['grid']), np.array(ra['y']['values'])
    gz, vz = np.array(ra['z']['grid']), np.array(ra['z']['values'])
    L = float(np.sqrt(np.diff(vx) ** 2 + np.diff(vy) ** 2 + np.diff(vz) ** 2).sum())
    tw = d['components']['blade']['outer_shape']['twist']
    gt, vt = np.array(tw['grid']), np.array(tw['values'])
    return (gx, vx), (gy, vy), (gt, vt), L


def load_ff():
    """16-station (eta, P_perim, x2bar, x2ref) from ff_beam_load.txt (rows are sorted by eta)."""
    rows = []
    for line in open(FFTXT):
        s = line.split()
        if len(s) != 9:
            continue
        try:
            rows.append([float(x) for x in s])
        except ValueError:
            continue
    ff = np.array(rows)                       # (16, 9): eta span chord P ds Fx x2bar x2ref Mz
    assert ff.shape[0] == 16, 'expected 16 ff rows, got %d' % ff.shape[0]
    return ff[:, 3], ff[:, 6], ff[:, 7]       # P_perim, x2bar, x2ref


# ================================================================== PROPS
K, M = load_KM()
Kb, Mb = transformMatrixToBeamDyn(K.copy(), M.copy())
_pf = write_beamdyn_prop(HERE, 'iea', ETAS, Kb, Mb, [1e-3] * 6)   # user's writer -> bd_props_iea.inp
os.replace(os.path.join(HERE, _pf), os.path.join(HERE, 'iea_bd_props.inp'))
print('wrote iea_bd_props.inp   (station_total=16, VABS->BeamDyn remap)')

# ================================================================== PRIMARY
# STRAIGHT + UNTWISTED reference line (match BAR-URC): the OpenSG 6x6 is computed in the local
# section frame with no prebend/twist context, so the beam model must add none.  Only kp_zr varies.
(gx, vx), (gy, vy), (gt, vt), L = windio_axes()
NKP = int(round(L / 2.0))                       # RULE: kp_total = round(blade_length / 2)
eta_kp = np.linspace(0.0, 1.0, NKP)
kp_xr = np.zeros(NKP)                          # no prebend
kp_yr = np.zeros(NKP)                          # no offset
kp_zr = eta_kp * L                             # spanwise [m]
kp_tw = np.zeros(NKP)                          # no twist

src = open(os.path.join(BARURC, 'bd_primary_bar_urc.inp')).read().splitlines()
out = []
i = 0
while i < len(src):
    line = src[i]
    if line.strip().startswith('kp_xr'):
        out.append(line)                      # kp_xr / kp_yr / kp_zr / initial_twist header
        out.append(src[i + 1])                # (m) (m) (m) (deg) header
        for k in range(NKP):
            out.append('\t %13.5e \t %13.5e \t %13.5e \t %13.5e ' %
                        (kp_xr[k], kp_yr[k], kp_zr[k], kp_tw[k]))
        j = i + 2
        while 'MESH PARAMETER' not in src[j]:
            j += 1
        i = j
        continue
    if 'bar_urc blade' in line:
        out.append('iea_22 blade'); i += 1; continue
    # NOTE on quadrature: trapezoidal (quadrature=2) was tried so the quadrature points would sit on
    # the 16 station etas, but BeamDyn's BldNd force output is ALWAYS at the order_elem+1 GLL FE nodes
    # (the .sum confirmed 11 output nodes even with trapezoidal) AND the static solve did not converge
    # for this soft-tip blade.  So we keep Gaussian (quadrature=1, converges) and cubic-interpolate the
    # smooth 11-node FF onto the 16 airfoil etas in extract_ff.py (coordinator-approved fallback).
    if 'kp_total' in line and 'Total number of key points' in line:
        out.append('%11d   kp_total        - Total number of key points (-) [must be at least 3]' % NKP)
        i += 1; continue
    if 'Member number; Number of key points' in line:
        out.append('     1     %d                 - Member number; Number of key points in this member' % NKP)
        i += 1; continue
    if 'bd_props_bar_urc.inp' in line:
        out.append(line.replace('bd_props_bar_urc.inp', 'iea_bd_props.inp')); i += 1; continue
    out.append(line); i += 1
open(os.path.join(HERE, 'iea_bd_primary.inp'), 'w').write('\n'.join(out) + '\n')
print('wrote iea_bd_primary.inp (L=%.3f m, kp_total=%d [=round(L/2)], STRAIGHT & UNTWISTED '
      'kp_xr=kp_yr=twist=0, Gaussian quadrature)' % (L, NKP))

# ================================================================== DRIVER
P16, xbar16, xref16 = load_ff()
eta_pl = np.round(np.arange(0.0, 1.0 + 1e-9, 0.02), 4)      # 0,0.02,...,1.0 -> 51 pts
assert len(eta_pl) == NPL, len(eta_pl)
P_pl = np.interp(eta_pl, ETAS, P16)
xbar_pl = np.interp(eta_pl, ETAS, xbar16)
xref_pl = np.interp(eta_pl, ETAS, xref16)
deta = 0.02
ds = np.full(NPL, deta * L)
ds[0] = ds[-1] = 0.5 * deta * L                            # end nodes carry half a tributary span
Fx = P_PA * P_pl * ds                                      # flapwise point force [N]
Mz = Fx * (xbar_pl - xref_pl)                              # offset torsion [N-m]

src = open(os.path.join(BARURC, 'bd_bar_urc.inp')).read().splitlines()
out = []
i = 0
while i < len(src):
    line = src[i]
    if 'Static analysis of a twisted beam' in line:
        out.append('Static analysis of the IEA-22 blade under 1200 Pa flapwise surface traction')
        i += 1; continue
    if 'NumPointLoads' in line:
        out.append('%11d   NumPointLoads - Number of point loads along blade' % NPL)
        out.append(src[i + 1])                # "Non-dim blade-span eta   Fx ..." header
        out.append(src[i + 2])                # "(-)  (N) ..." units header
        for k in range(NPL):
            out.append('%.4f %.10g %.1f %.1f %.1f %.1f %.10g'
                        % (eta_pl[k], Fx[k], 0.0, 0.0, 0.0, 0.0, Mz[k]))
        j = i + 3
        while 'PRIMARY INPUT FILE' not in src[j]:
            j += 1
        i = j
        continue
    if 'bd_primary_bar_urc.inp' in line:
        out.append(line.replace('bd_primary_bar_urc.inp', 'iea_bd_primary.inp')); i += 1; continue
    out.append(line); i += 1
open(os.path.join(HERE, 'iea_bd_driver.inp'), 'w').write('\n'.join(out) + '\n')
print('wrote iea_bd_driver.inp  (p=%.0f Pa, %d point loads, sum Fx=%.1f N = %.4f MN, max Mz=%.3e N-m)'
      % (P_PA, NPL, Fx.sum(), Fx.sum() / 1e6, np.abs(Mz).max()))
print('done.')
