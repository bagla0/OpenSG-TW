'''shift_1d_refaxis.py -- move the (validated, committed) mid-surface 1-D shell YAMLs from the
LE origin to the windIO REFERENCE-AXIS origin by a pure rigid translation: subtract section_offset_y(eta)
from every node x2.  Only the node coordinates change (elements, orientations, sets, materials
untouched), so EA/GA/GJ are unchanged (still validated +/-1% vs the 2-D solid) and only EI2/EI3 move to
the reference axis -- consistent with ff_beam_load.  Edits 1d_yaml/iea_r*_shell.yaml in place.
    ~/miniconda3/envs/opensg_2_0/bin/python shift_1d_refaxis.py
'''
import glob
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
YDIR = os.path.join(HERE, '1d_yaml')
ETA = {'r0000': 0.0, 'r0020': 0.02, 'r0049': 0.04868313468735217, 'r0066': 0.06649308419703373,
       'r0083': 0.08345591801468194, 'r0102': 0.10220904193570582, 'r0110': 0.11036272014164386,
       'r0136': 0.1364246061019374, 'r0156': 0.15564440587515185, 'r0197': 0.19665336575444797,
       'r0247': 0.24696148735364706, 'r0399': 0.3992636115637571, 'r0534': 0.5335887750152993,
       'r0739': 0.738938689884722, 'r0980': 0.9799991709122947, 'r1000': 1.0}

_d = yaml.safe_load(open(os.path.join(HERE, 'IEA-22-280-RWT.yaml')))
_so = _d['components']['blade']['outer_shape']['section_offset_y']
SO_G, SO_V = np.array(_so['grid']), np.array(_so['values'])


def section_offset_y(e):
    return float(np.interp(e, SO_G, SO_V))


for f in sorted(glob.glob(os.path.join(YDIR, 'iea_r*_shell.yaml'))):
    tag = os.path.basename(f).split('_')[1]
    if tag not in ETA:
        continue
    so = section_offset_y(ETA[tag])
    lines = open(f).read().splitlines()
    out, in_nodes = [], False
    xs = []
    for ln in lines:
        s = ln.strip()
        if s == 'nodes:':
            in_nodes = True; out.append(ln); continue
        if in_nodes and not s.startswith('-'):        # reached 'elements:' (or next key)
            in_nodes = False
        if in_nodes and s.startswith('-'):
            body = s[1:].strip().strip('[]')
            t = body.split()
            x = float(t[0]) - so; xs.append(x)
            out.append('- [%.8f %.8f %.8f]' % (x, float(t[1]), float(t[2])))
        else:
            out.append(ln)
    with open(f, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print('%-6s offy=%.3f  x2 in [%.3f, %.3f]  (ref axis at 0)' % (tag, so, min(xs), max(xs)))
print('shifted 1d_yaml to reference-axis origin (validated mid-surface geometry preserved)')
