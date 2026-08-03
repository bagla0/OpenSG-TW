'''frame + spar-cap check for the r0.2 station (iea_s10) across solid yaml / shell yaml / .sg.'''
import os, numpy as np, yaml
ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'


def rowf(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


def yaml_bbox(p, label):
    if not os.path.exists(p):
        print('  %-26s MISSING (%s)' % (label, p)); return None
    d = yaml.safe_load(open(p))
    nd = np.array([rowf(n)[:2] for n in d['nodes']])
    els = d['elements']
    elens = sorted(set(len(rowf(e)) for e in els))
    print('  %-26s nodes=%d elems=%d elens=%s  x[%.4f,%.4f] y[%.4f,%.4f]'
          % (label, len(nd), len(els), elens, nd[:, 0].min(), nd[:, 0].max(), nd[:, 1].min(), nd[:, 1].max()))
    if 'sets' in d:
        sets = d['sets']['element'] if isinstance(d['sets'], dict) else d['sets']
        print('       material sets: %s' % [s['name'] for s in sets])
    if 'sections' in d:
        print('       shell sections: %d' % len(d['sections']))
    return d


print('=== r0.2 station iea_s10 frames ===')
yaml_bbox(os.path.join(ROOT, 'shell51/2d_hybrid/iea_s10_solid.yaml'), '2d_hybrid solid')
yaml_bbox(os.path.join(ROOT, 'shell51/fallback_yaml/iea_s10_solid.yaml'), 'fallback solid')
yaml_bbox(os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml'), '1d shell')

# .sg bbox
sg = os.path.join(ROOT, 'shell51/sg_v201/iea_s10.sg')
if os.path.exists(sg):
    L = [l for l in open(sg).read().splitlines() if l.strip()]
    h = next(i for i, l in enumerate(L) if len(l.split()) == 3 and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
    nn = int(L[h].split()[0])
    xy = np.array([[float(L[h + 1 + k].split()[1]), float(L[h + 1 + k].split()[2])] for k in range(nn)])
    print('  %-26s nodes=%d  x[%.4f,%.4f] y[%.4f,%.4f]' % ('sg_v201 .sg (VABS)', nn, xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()))

# any VABS .SM / .U for r0.2 on the server?
print('\n=== VABS .SM/.U for r0.2 present on server? ===')
import glob
for pat in ['**/iea_s10*.SM', '**/iea_r020*.SM', '**/iea_s10*.U', '**/*r020*.SM']:
    for f in glob.glob(os.path.join(ROOT, '..', pat), recursive=True)[:4]:
        print('   ', f)
