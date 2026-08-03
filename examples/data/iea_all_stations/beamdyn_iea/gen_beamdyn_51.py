'''gen_beamdyn_51.py  --  build BeamDyn input files for the IEA-22 blade from the 51 OpenSG
cross-sections (Timoshenko 6x6 stiffness K + 6x6 mass M), and run beamdyn_driver.

TWO SEPARATE BeamDyn models are produced (the shell-vs-solid comparison at the beam level):
    solid : shell51/out/OpenSG_Hybrid_Solid/iea_sNN_OpenSG_Hybrid_Solid.out   (2-D solid, PreVABS tri + refined quad)
    shell : shell51/out/OpenSG_RM_Shell/iea_sNN_OpenSG_RM_Shell.out            (RM 1-D shell ring)
Each -> its own {prefix}_bd_props.inp (blade sectional K,M) + {prefix}_bd_primary.inp (reference
line) + {prefix}_bd_driver.inp (loads); beamdyn_driver reads the driver and writes {prefix}_bd_driver.out.

--------------------------------------------------------------------------------------------------
CROSS-SECTION DATA .out (VABS/OpenSG order  1=extension, 2,3=transverse shear, 4=torsion, 5,6=bending):
  read the 6x6 "Stiffness" block and the 6x6 "The 6X6 Mass Matrix" block; both referenced to the
  windIO reference axis x1.  BeamDyn uses a different axis convention, so each 6x6 is rotated by the
  VABS->BeamDyn frame map  B = [[0,0,1],[0,-1,0],[1,0,0]]  (beamdyn_trans.transformMatrixToBeamDyn):
      M_bd = T^T M T   applied block-wise (3x3 blocks),   T = B^{-1}.

--------------------------------------------------------------------------------------------------
FORCE MODEL  --  distributed surface traction p [Pa] -> per-station beam load, taken ABOUT THE x1
REFERENCE AXIS with the moment arm measured FROM the x1 axis (NOT the LE):

  Geometry: OML airfoil = PreVABS .dat (normalized 0..1) x chord [m], then shifted chordwise so the
  windIO reference axis is the origin:   x2 <- x2 - section_offset_y(eta)   (LE->ref-axis distance).
  This x1 axis is the SAME origin the 6x6 is referenced to, so force and stiffness share one frame.

  Per OML line element k (nodes a,b), in metres, about x1:
      ds_k  = || r_b - r_a ||                         element arc length            [m]
      x2_k  = 0.5 (x2_a + x2_b)                        chordwise moment arm FROM x1  [m]   (signed)
      P_i   = sum_k ds_k                               section perimeter             [m]
  Per station i with tributary (bin) span ds_bin_i along the span (end stations carry a half-bin):
      F_flap_i = p * P_i               * ds_bin_i      flapwise force  (out-of-plane, +x3)  [N]
      M_z_i    = p * ( sum_k x2_k ds_k )_i * ds_bin_i   torsion about x1 = integral of p*x2 over
                                                        the perimeter (force x moment-arm)   [N-m]
      F_chord_i = 0
  i.e. dF = p ds ds_bin ;  dM_z = p x2 ds ds_bin = dF * (moment arm x2 from x1).

Static solve, gravity off.
    ~/miniconda3/envs/opensg_2_0/bin/python gen_beamdyn_51.py
'''
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))            # .../beamdyn_iea
ROOT = os.path.dirname(HERE)                                 # .../iea_all_stations
SHELL51 = os.path.join(ROOT, 'shell51')
OUTROOT = os.path.join(SHELL51, 'out')
XMLDIR = os.path.join(SHELL51, 'xml')
BARURC = os.path.join(ROOT, 'beamdyn_barurc')
WINDIO = os.path.join(ROOT, 'IEA-22-280-RWT.yaml')

sys.path.insert(0, HERE)
from beamdyn_trans import transformMatrixToBeamDyn, write_beamdyn_prop

P_PA = 1500.0                          # flapwise surface traction [Pa]
NSTA = 51                              # 51 cross-sections, eta_i = i/50
ETAS = np.arange(NSTA) / 50.0
TAGS = ['iea_s%02d' % i for i in range(NSTA)]
SOURCES = {                            # 2 separate BeamDyn models: (out-subfolder, file prefix)
    'solid': ('OpenSG_Hybrid_Solid', 'iea51_solid'),
    'shell': ('OpenSG_RM_Shell', 'iea51_shell'),
}

# ------------------------------------------------------------------ windIO reference axis / offset
_d = yaml.safe_load(open(WINDIO))
_bl = _d['components']['blade']
_so = _bl['outer_shape']['section_offset_y']
SO_G, SO_V = np.array(_so['grid'], float), np.array(_so['values'], float)
_ra = _bl['reference_axis']
_rx, _ry, _rz = (np.array(_ra[k]['values'], float) for k in ('x', 'y', 'z'))
L = float(np.sqrt(np.diff(_rx) ** 2 + np.diff(_ry) ** 2 + np.diff(_rz) ** 2).sum())   # blade length [m]


def section_offset_y(e):
    return float(np.interp(e, SO_G, SO_V))                  # LE -> reference-axis chordwise distance [m]


# ------------------------------------------------------------------ read K, M from a .out
def parse_out(path):
    '''Read the 6x6 "Stiffness" and 6x6 "The 6X6 Mass Matrix" blocks (VABS/OpenSG order).'''
    lines = open(path).read().splitlines()

    def read6(start):                                       # next 6 rows with >=6 floats from `start`
        rows, j = [], start
        while len(rows) < 6 and j < len(lines):
            try:
                v = [float(x) for x in lines[j].split()]
                if len(v) >= 6:
                    rows.append(v[:6])
            except ValueError:
                pass
            j += 1
        return np.array(rows)

    K = M = None
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith('Stiffness'):
            K = read6(i + 1)
        elif s.lower().startswith('the 6x6 mass matrix'):
            M = read6(i + 1)
    if K is None or M is None or K.shape != (6, 6) or M.shape != (6, 6):
        raise ValueError('could not parse K/M from %s' % path)
    return K, M


# ------------------------------------------------------------------ OML force about x1
def read_dat(path):
    pts = []
    for ln in open(path):
        s = ln.split()
        if len(s) == 2:
            try:
                pts.append([float(s[0]), float(s[1])])
            except ValueError:
                pass
    return np.array(pts)


def oml_force(xmlpath, eta):
    '''Return (P, sum_x2ds, chord) about x1 for one station (see FORCE MODEL in the header).'''
    root = ET.parse(xmlpath).getroot()
    chord = float(root.find('general/scale').text)                      # chord [m]
    datname = root.find(".//line[@type='airfoil']/points").text.strip()
    xy = read_dat(os.path.join(XMLDIR, datname)) * chord                 # metres, LE origin
    xy[:, 0] -= section_offset_y(eta)                                    # shift origin onto x1 axis
    closed = np.vstack([xy, xy[0]])                                      # close the OML loop
    seg = np.diff(closed, axis=0)
    ds = np.hypot(seg[:, 0], seg[:, 1])                                  # ds_k  (element arc length)
    x2mid = 0.5 * (closed[:-1, 0] + closed[1:, 0])                       # x2_k  (moment arm from x1)
    return float(ds.sum()), float((x2mid * ds).sum()), chord


def station_loads():
    '''Per-station F_flap and M_z about x1 (identical for the solid and shell beam models,
    since the load is geometry-driven).  Returns eta, F_flap[N], M_z[N-m], and a printable table.'''
    # tributary bin span: midpoints between the uniform station etas, ends get a half bin
    b = np.concatenate([[ETAS[0]], 0.5 * (ETAS[:-1] + ETAS[1:]), [ETAS[-1]]])
    ds_bin = np.diff(b) * L                                             # [m]
    Fflap = np.zeros(NSTA); Mz = np.zeros(NSTA); rows = []
    for i in range(NSTA):
        xmlp = os.path.join(XMLDIR, TAGS[i] + '.xml')
        P, Sx2ds, chord = oml_force(xmlp, ETAS[i])
        Fflap[i] = P_PA * P * ds_bin[i]                                 # F_flap = p*P*ds_bin
        Mz[i] = P_PA * Sx2ds * ds_bin[i]                               # M_z = p*sum(x2*ds)*ds_bin
        rows.append((TAGS[i], ETAS[i], ETAS[i] * L, chord, P, ds_bin[i], Fflap[i], Mz[i]))
    return ETAS, Fflap, Mz, rows


# ------------------------------------------------------------------ write the 3 BeamDyn files
def write_primary(prefix):
    '''STRAIGHT + UNTWISTED reference line along x1 (kp_xr=kp_yr=twist=0; kp_zr=eta*L). The section
    6x6 is in the local frame with no prebend context, so the beam adds none (matches gen_beamdyn_iea).'''
    NKP = int(round(L / 2.0))                                           # ~1 key point / 2 m span
    eta_kp = np.linspace(0.0, 1.0, NKP)
    kp = np.column_stack([np.zeros(NKP), np.zeros(NKP), eta_kp * L, np.zeros(NKP)])   # xr yr zr twist
    src = open(os.path.join(BARURC, 'bd_primary_bar_urc.inp')).read().splitlines()
    out, i = [], 0
    while i < len(src):
        line = src[i]
        # TRAPEZOIDAL quadrature (2): the quadrature/output nodes coincide with the 51 blade
        # property stations, so BeamDyn writes nodal output (N001..N051) AT every cross-section
        # (Gaussian(1) only outputs at the order_elem+1 GLL nodes -> 11 nodes for order 10).
        if 'quadrature' in line and 'Quadrature method' in line:
            out.append('          2   quadrature       - Quadrature method: 1=Gaussian; 2=Trapezoidal (switch)')
            i += 1; continue
        if 'refine' in line and 'Refinement factor' in line:
            out.append('          1   refine           - Refinement factor for trapezoidal quadrature (-) [DEFAULT = 1; used only when quadrature=2]')
            i += 1; continue
        if line.strip().startswith('kp_xr'):
            out.append(line); out.append(src[i + 1])                    # 2 header rows
            for k in range(NKP):
                out.append('\t %13.5e \t %13.5e \t %13.5e \t %13.5e ' % tuple(kp[k]))
            j = i + 2
            while 'MESH PARAMETER' not in src[j]:
                j += 1
            i = j; continue
        if 'bar_urc blade' in line:
            out.append('iea_22 blade (%s)' % prefix); i += 1; continue
        if 'kp_total' in line and 'Total number of key points' in line:
            out.append('%11d   kp_total        - Total number of key points (-) [must be at least 3]' % NKP)
            i += 1; continue
        if 'Member number; Number of key points' in line:
            out.append('     1     %d                 - Member number; Number of key points in this member' % NKP)
            i += 1; continue
        if 'bd_props_bar_urc.inp' in line:
            out.append(line.replace('bd_props_bar_urc.inp', '%s_bd_props.inp' % prefix)); i += 1; continue
        out.append(line); i += 1
    open(os.path.join(HERE, '%s_bd_primary.inp' % prefix), 'w').write('\n'.join(out) + '\n')
    return NKP


def write_driver(prefix, eta, Fflap, Mz):
    src = open(os.path.join(BARURC, 'bd_bar_urc.inp')).read().splitlines()
    out, i = [], 0
    while i < len(src):
        line = src[i]
        if 'Static analysis of a twisted beam' in line:
            out.append('Static analysis of the IEA-22 blade (%s) under %.0f Pa flapwise surface traction'
                       % (prefix, P_PA)); i += 1; continue
        if 'NumPointLoads' in line:
            out.append('%11d   NumPointLoads - Number of point loads along blade' % NSTA)
            out.append(src[i + 1]); out.append(src[i + 2])              # 2 header rows
            for k in range(NSTA):
                # columns: eta  Fx(flap)  Fy  Fz  Mx  My  Mz(torsion about x1)
                out.append('%.4f %.10g %.1f %.1f %.1f %.1f %.10g'
                           % (eta[k], Fflap[k], 0.0, 0.0, 0.0, 0.0, Mz[k]))
            j = i + 3
            while 'PRIMARY INPUT FILE' not in src[j]:
                j += 1
            i = j; continue
        if 'bd_primary_bar_urc.inp' in line:
            out.append(line.replace('bd_primary_bar_urc.inp', '%s_bd_primary.inp' % prefix)); i += 1; continue
        out.append(line); i += 1
    open(os.path.join(HERE, '%s_bd_driver.inp' % prefix), 'w').write('\n'.join(out) + '\n')


def build_model(source, eta, Fflap, Mz):
    sub, prefix = SOURCES[source]
    # 1. read + transform the 51 cross-section 6x6 K, M
    K = np.zeros((NSTA, 6, 6)); M = np.zeros((NSTA, 6, 6))
    for i in range(NSTA):
        p = os.path.join(OUTROOT, sub, '%s_%s.out' % (TAGS[i], sub))
        K[i], M[i] = parse_out(p)
    Kb, Mb = transformMatrixToBeamDyn(K.copy(), M.copy())               # VABS -> BeamDyn frame
    # 2. blade props (station K, M), primary (reference line), driver (loads)
    pf = write_beamdyn_prop(HERE, prefix, ETAS, Kb, Mb, [1e-3] * 6)
    os.replace(os.path.join(HERE, pf), os.path.join(HERE, '%s_bd_props.inp' % prefix))
    nkp = write_primary(prefix)
    write_driver(prefix, eta, Fflap, Mz)
    print('  [%-5s] wrote %s_bd_{props,primary,driver}.inp  (51 stations, kp_total=%d)'
          % (source, prefix, nkp))


def main():
    eta, Fflap, Mz, rows = station_loads()
    print('IEA-22 blade  L=%.3f m  p=%.0f Pa  (51 stations, loads about x1 reference axis)' % (L, P_PA))
    print('%-8s %6s %8s %7s %8s %8s %13s %13s' %
          ('tag', 'eta', 'span[m]', 'chord', 'P_OML', 'ds_bin', 'F_flap[N]', 'M_z[N-m]'))
    for r in rows:
        print('%-8s %6.3f %8.2f %7.3f %8.3f %8.3f %13.1f %13.2f' % r)
    print('  sum F_flap = %.1f N (%.4f MN)   sum M_z = %.4e N-m' % (Fflap.sum(), Fflap.sum() / 1e6, Mz.sum()))
    print('\nbuilding 2 BeamDyn models (solid + shell):')
    for source in ('solid', 'shell'):
        build_model(source, eta, Fflap, Mz)
    print('done. run:  beamdyn_driver iea51_solid_bd_driver.inp   and   iea51_shell_bd_driver.inp')


if __name__ == '__main__':
    main()
