'''ff_beam_load.py -- 1500 Pa flapwise surface traction -> per-station beam load (BeamDyn/GEBT).

GEOMETRY = the OML airfoil, taken from the PreVABS xml input (xml/iea_r*.xml + its .dat), which is
the canonical OML: the xml `<baselines><line type="airfoil">` points to the .dat (normalized OML,
0=LE..1=TE) and `<general><scale>` is the chord.  The webs are SEPARATE xml components
(<component name="web_*">), so the .dat is OML-only -- no web filtering needed.

ORIGIN = the windIO REFERENCE AXIS (a general internal point, ~1/3 chord), NOT the leading edge.
The chordwise LE->reference-axis distance is windIO outer_shape.section_offset_y(eta) [m]; we subtract
it so x2=0 sits on the reference axis.  Stacking every section at this point gives a STRAIGHT x1 beam
line (the LE would zig-zag).  This is the same origin the RM 6x6 must be referenced to.

Per OML line element k (nodes a,b) after scaling to metres and shifting to the reference axis:
    ds_k  = || node_b - node_a ||                                (euclidean element length)
    x2_k  = 0.5(x2_a + x2_b)   with x2 measured FROM THE REFERENCE AXIS   (= the moment arm)
    P     = sum ds_k                                             (OML perimeter)
Then per station with tributary bin span ds_bin:
    F_flap = p * P      * ds_bin                                 flap force (+x3)
    M_z    = p * (sum x2_k ds_k) * ds_bin                        torsion about the reference axis x1
    F_chord = 0
    ~/miniconda3/envs/opensg_2_0/bin/python ff_beam_load.py
'''
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WINDIO = os.path.join(HERE, 'IEA-22-280-RWT.yaml')
XMLDIR = os.path.join(HERE, 'xml')
P_PA = 1500.0                                              # flapwise surface traction [Pa]

ETA = {'r0000': 0.0, 'r0020': 0.02, 'r0049': 0.04868313468735217, 'r0066': 0.06649308419703373,
       'r0083': 0.08345591801468194, 'r0102': 0.10220904193570582, 'r0110': 0.11036272014164386,
       'r0136': 0.1364246061019374, 'r0156': 0.15564440587515185, 'r0197': 0.19665336575444797,
       'r0247': 0.24696148735364706, 'r0399': 0.3992636115637571, 'r0534': 0.5335887750152993,
       'r0739': 0.738938689884722, 'r0980': 0.9799991709122947, 'r1000': 1.0}

_d = yaml.safe_load(open(WINDIO))
_osh = _d['components']['blade']['outer_shape']
_so = _osh['section_offset_y']; _ra = _d['components']['blade']['reference_axis']
SO_G, SO_V = np.array(_so['grid']), np.array(_so['values'])
_rx, _ry, _rz = (np.array(_ra[k]['values']) for k in ('x', 'y', 'z'))
L = float(np.sqrt(np.diff(_rx) ** 2 + np.diff(_ry) ** 2 + np.diff(_rz) ** 2).sum())


def section_offset_y(e):
    return float(np.interp(e, SO_G, SO_V))                # LE -> reference-axis chordwise distance [m]


def read_dat(path):
    '''OML airfoil points (normalized 0..1), skipping the header line.'''
    pts = []
    for ln in open(path):
        s = ln.split()
        if len(s) == 2:
            try:
                pts.append([float(s[0]), float(s[1])])
            except ValueError:
                pass
    return np.array(pts)


def oml_from_xml(xmlpath, e):
    '''Return (P, sum_x2ds, chord, section_offset_y, x2min, x2max, n_web) for the OML about the
    reference axis. Geometry from the xml-referenced .dat (OML); webs are separate xml components.'''
    root = ET.parse(xmlpath).getroot()
    scale = float(root.find('general/scale').text)        # chord [m]
    datname = root.find(".//line[@type='airfoil']/points").text.strip()
    n_web = len([c for c in root.findall('component') if (c.get('name') or '').startswith('web')])
    xy = read_dat(os.path.join(XMLDIR, datname)) * scale   # metres, LE origin
    so = section_offset_y(e)
    xy[:, 0] -= so                                          # shift origin to the reference axis
    closed = np.vstack([xy, xy[0]])                        # close the OML (TE gap)
    seg = np.diff(closed, axis=0)
    ds = np.hypot(seg[:, 0], seg[:, 1])
    x2mid = 0.5 * (closed[:-1, 0] + closed[1:, 0])
    return (float(ds.sum()), float((x2mid * ds).sum()), scale, so,
            float(xy[:, 0].min()), float(xy[:, 0].max()), n_web)


def main():
    files = sorted(glob.glob(os.path.join(XMLDIR, 'iea_r*.xml')))
    tag = [os.path.basename(f)[:-4].split('_')[-1] for f in files]      # 'r0247'
    keep = [(t, f) for t, f in zip(tag, files) if t in ETA]
    keep.sort(key=lambda tf: ETA[tf[0]])
    tag = [t for t, _ in keep]; files = [f for _, f in keep]
    eta = np.array([ETA[t] for t in tag])

    b = np.concatenate([[eta[0]], 0.5 * (eta[:-1] + eta[1:]), [eta[-1]]])
    dss = np.diff(b) * L                                                # tributary (bin) span [m]

    rows = []
    for i, f in enumerate(files):
        P, Sx2ds, chord, so, x2lo, x2hi, nweb = oml_from_xml(f, eta[i])
        Fflap = P_PA * P * dss[i]
        Mz = P_PA * Sx2ds * dss[i]
        rows.append((tag[i], eta[i], eta[i] * L, chord, so, P, nweb, dss[i], Fflap, Sx2ds, Mz, x2lo, x2hi))

    hdr = ('%-7s %7s %8s %7s %8s %8s %4s %8s %13s %10s %13s %8s %8s'
           % ('tag', 'eta', 'span[m]', 'chord', 'offy', 'P_OML', 'nweb', 'ds_bin', 'Fflap[N]',
              'Sx2ds', 'Mz[N-m]', 'x2min', 'x2max'))
    print(hdr)
    for r in rows:
        print('%-7s %7.4f %8.2f %7.3f %8.3f %8.3f %4d %8.3f %13.1f %10.3f %13.2f %8.3f %8.3f' % r)
    sF = sum(r[8] for r in rows); sM = sum(r[10] for r in rows)
    print('\nL=%.2f m  p=%.0f Pa  |  sum Fflap=%.1f N (=%.3f MN)  |  sum Mz=%.4e N-m' % (L, P_PA, sF, sF / 1e6, sM))
    print('geometry = OML from xml/.dat; origin = windIO reference axis (x2=0), LE at x2=-offy, TE at +.')

    with open(os.path.join(HERE, 'ff_beam_load.txt'), 'w') as fh:
        fh.write(__doc__ + '\n' + hdr + '\n')
        for r in rows:
            fh.write('%-7s %7.4f %8.2f %7.3f %8.3f %8.3f %4d %8.3f %13.1f %10.3f %13.2f %8.3f %8.3f\n' % r)
        fh.write('\nL=%.3f m  p=%.0f Pa  sum Fflap=%.1f N  sum Mz=%.4e N-m\n' % (L, P_PA, sF, sM))
    np.savetxt(os.path.join(HERE, 'ff_beam_load.dat'),
               np.column_stack([[r[1] for r in rows], [r[2] for r in rows],
                                [r[8] for r in rows], [r[10] for r in rows]]),
               header='eta  span[m]  Fflap[N]  Mz[N-m]  (p=1500Pa flapwise; OML from xml; about windIO reference axis)',
               fmt='%.8e')
    print('wrote ff_beam_load.txt, ff_beam_load.dat')


if __name__ == '__main__':
    main()
