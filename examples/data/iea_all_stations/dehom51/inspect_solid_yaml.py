'''inspect the (0,0)-origin 2D solid yaml structure + confirm 1D shell shares the frame.'''
import os, numpy as np, yaml
ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
SOL = os.path.join(ROOT, 'shell51/2d_yaml/iea_s10_solid.yaml')
SHELL = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')


def row(v):
    if isinstance(v, str):
        return [float(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [float(x) for x in v[0].split()]
    return [float(x) for x in v]


y = yaml.safe_load(open(SOL))
print('TOP-LEVEL KEYS:', list(y.keys()))
nd = np.array([row(n)[:2] for n in y['nodes']])
print('nodes: %d   x[%.3f, %.3f]  y[%.3f, %.3f]  centroid=(%.3f, %.3f)'
      % (len(nd), nd[:, 0].min(), nd[:, 0].max(), nd[:, 1].min(), nd[:, 1].max(), nd[:, 0].mean(), nd[:, 1].mean()))
print('elements: %d   elem[0]=%s' % (len(y['elements']), y['elements'][0]))
for k in y.keys():
    if k in ('nodes', 'elements'):
        continue
    v = y[k]
    n = len(v) if hasattr(v, '__len__') else v
    print('--- %s (len=%s, type=%s)' % (k, n, type(v).__name__))
    if isinstance(v, dict):
        for kk in list(v.keys())[:6]:
            vv = v[kk]
            print('     %-24s : %s' % (kk, (str(vv)[:90] if not hasattr(vv, '__len__') or isinstance(vv, str) else 'len=%d  head=%s' % (len(vv), str(vv[:2])[:90]))))
    elif isinstance(v, list):
        print('     head:', str(v[:3])[:200])

# 1D shell frame
sh = yaml.safe_load(open(SHELL))
snd = np.array([row(n)[:2] for n in sh['nodes']])
print('\n1D SHELL nodes: %d   x[%.3f, %.3f]  y[%.3f, %.3f]' % (len(snd), snd[:, 0].min(), snd[:, 0].max(), snd[:, 1].min(), snd[:, 1].max()))
print('==> both at (0,0) ref-axis?  solid_minx=%.3f  shell_minx=%.3f  (LE at negative x)' % (nd[:, 0].min(), snd[:, 0].min()))
