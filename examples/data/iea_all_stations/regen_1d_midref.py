'''regen_1d_midref.py -- regenerate the 16 1-D shell YAMLs on the laminate MID-surface (center
reference, validated to +/-1% vs the 2-D solid) with the in-plane origin at the windIO REFERENCE
AXIS (x1), not the LE:
  * mid-surface contour : emit_opensg_yaml(cs, ..., fraction=0.5)   (per-node inward offset = center ref)
  * reference-axis origin: shift cs['nodes'].x by -section_offset_y(eta)  (rigid translation)
Overwrites 1d_yaml/iea_r*_shell.yaml (the canonical RM input).  Also prints the windIO global-frame
data (chord, section_offset_y, reference_axis) so the origin convention is explicit.
    ~/miniconda3/envs/opensg_2_0/bin/python regen_1d_midref.py
'''
import glob
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser('~/OpenSG-TW-claude/third_party/OpenSG_io'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from opensg_io.prevabs_xml import parse_prevabs_xml
from opensg_io.converter import emit_opensg_yaml

XMLDIR = os.path.join(HERE, 'xml')
OUT = os.path.join(HERE, '1d_yaml'); os.makedirs(OUT, exist_ok=True)
ETA = {'r0000': 0.0, 'r0020': 0.02, 'r0049': 0.04868313468735217, 'r0066': 0.06649308419703373,
       'r0083': 0.08345591801468194, 'r0102': 0.10220904193570582, 'r0110': 0.11036272014164386,
       'r0136': 0.1364246061019374, 'r0156': 0.15564440587515185, 'r0197': 0.19665336575444797,
       'r0247': 0.24696148735364706, 'r0399': 0.3992636115637571, 'r0534': 0.5335887750152993,
       'r0739': 0.738938689884722, 'r0980': 0.9799991709122947, 'r1000': 1.0}

d = yaml.safe_load(open(os.path.join(HERE, 'IEA-22-280-RWT.yaml')))
osh = d['components']['blade']['outer_shape']; ra = d['components']['blade']['reference_axis']
SO_G, SO_V = np.array(osh['section_offset_y']['grid']), np.array(osh['section_offset_y']['values'])
CH_G, CH_V = np.array(osh['chord']['grid']), np.array(osh['chord']['values'])
RX = {k: (np.array(ra[k]['grid']), np.array(ra[k]['values'])) for k in ('x', 'y', 'z')}


def interp(gv, e):
    return float(np.interp(e, gv[0], gv[1]))


def section_offset_y(e):
    return float(np.interp(e, SO_G, SO_V))


print('windIO global frame:  reference_axis = beam x1;  section_offset_y = LE->refaxis chordwise dist')
print('%-6s %6s %8s %8s %8s | %8s %8s %9s' % ('tag', 'eta', 'chord', 'offy', 'offy/c',
                                              'refX(pre)', 'refY', 'refZ(span)'))
for tag in sorted(ETA, key=lambda t: ETA[t]):
    e = ETA[tag]
    print('%-6s %6.4f %8.3f %8.3f %8.3f | %8.3f %8.3f %9.3f'
          % (tag, e, interp((CH_G, CH_V), e), section_offset_y(e),
             section_offset_y(e) / interp((CH_G, CH_V), e),
             interp(RX['x'], e), interp(RX['y'], e), interp(RX['z'], e)))

print('\nregenerating 1d_yaml (mid-surface, reference-axis origin):')
for xml in sorted(glob.glob(os.path.join(XMLDIR, 'iea_r*.xml'))):
    tag = os.path.basename(xml)[:-4].split('_')[-1]
    if tag not in ETA:
        continue
    so = section_offset_y(ETA[tag])
    cs = parse_prevabs_xml(xml)
    cs['nodes'] = [np.array([float(p[0]) - so, float(p[1])]) for p in cs['nodes']]   # ref-axis origin
    out = os.path.join(OUT, 'iea_%s_shell.yaml' % tag)
    emit_opensg_yaml(cs, out, fraction=0.5)                                          # mid-surface
    nd = np.array([[float(x) for x in (r if isinstance(r, str) else r[0]).split()][:2]
                   for r in yaml.safe_load(open(out))['nodes']])
    print('  %-6s offy=%.3f  mid nodes x2 in [%.3f, %.3f]  (ref axis at 0)'
          % (tag, so, nd[:, 0].min(), nd[:, 0].max()))
print('wrote 1d_yaml/ (mid-surface + reference-axis origin)')
