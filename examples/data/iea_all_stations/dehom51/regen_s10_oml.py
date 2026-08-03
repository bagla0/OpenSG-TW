'''regen_s10_oml.py -- rebuild the r=0.2 (iea_s10) 1-D shell yaml at the OML reference
(fraction=0.0, no mid-surface inward offset) with the origin at the windIO reference axis,
exactly like regen_1d_oml.py does for the 16-station set.  so from the .refaxis sidecar.

Output -> shell51/1d_yaml_oml/iea_s10_shell.yaml   (reference: oml; old center yaml untouched)
'''
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, '..'))
IO_CANDS = [os.path.expanduser('~/OpenSG-TW-claude/third_party/OpenSG_io'),
            r'Y:\OpenSG-TW-claude\third_party\OpenSG_io',
            r'C:\Users\bagla0\OpenSG_io']
for c in IO_CANDS:
    if os.path.isdir(c):
        sys.path.insert(0, c)
        break
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from opensg_io.prevabs_xml import parse_prevabs_xml
from opensg_io.converter import emit_opensg_yaml

XML = os.path.join(IEA, 'shell51', 'xml', 'iea_s10.xml')
OUT = os.path.join(IEA, 'shell51', '1d_yaml_oml')
os.makedirs(OUT, exist_ok=True)

# reference-axis offset recorded when the center yaml was finalized
side = os.path.join(IEA, 'shell51', '1d_yaml', 'iea_s10_shell.yaml.refaxis')
so = float(open(side).read().split()[0].split('=')[1])
print('section_offset_y =', so)

cs = parse_prevabs_xml(XML)
cs['nodes'] = [np.array([float(p[0]) - so, float(p[1])]) for p in cs['nodes']]
out = os.path.join(OUT, 'iea_s10_shell.yaml')
emit_opensg_yaml(cs, out, fraction=0.0)

d = yaml.safe_load(open(out))
nd = np.array([[float(x) for x in (r if isinstance(r, str) else r[0]).split()][:2]
               for r in d['nodes']])
print('reference field :', d.get('reference'))
print('OML nodes x2 in [%.3f, %.3f], x3 in [%.3f, %.3f]  (ref axis at 0)'
      % (nd[:, 0].min(), nd[:, 0].max(), nd[:, 1].min(), nd[:, 1].max()))
print('wrote', out)
