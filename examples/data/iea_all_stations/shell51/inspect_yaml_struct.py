'''Print the structure of a 2d solid yaml: top-level keys and a sample of sets / sections /
materials / elementOrientations so we know how to remap after a mesh repair.'''
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '2d_yaml', 'iea_s28_solid.yaml')
d = yaml.safe_load(open(p))
print('top-level keys:', list(d.keys()))
for k in d:
    v = d[k]
    n = len(v) if hasattr(v, '__len__') else 'scalar'
    print('\n=== %s  (len=%s) ===' % (k, n))
    if k in ('nodes', 'elements'):
        print('  first 2:', v[:2])
        continue
    if isinstance(v, list):
        for item in v[:3]:
            print('  item:', repr(item)[:300])
    elif isinstance(v, dict):
        for kk in list(v.keys())[:6]:
            vv = v[kk]
            print('  %s: %s' % (kk, repr(vv)[:200]))
    else:
        print('  ', repr(v)[:200])
