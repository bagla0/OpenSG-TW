import numpy as np, yaml, os
ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
Y = ROOT + '/shell51/2d_hybrid/iea_s50_solid.yaml'

d = yaml.safe_load(open(Y))
def row(x): return [float(v) for v in (x[0].split() if isinstance(x, list) else str(x).split())]
nodes = np.array([row(n) for n in d['nodes']])
els = d['elements']
elens = sorted(set(len(row(e)) for e in els))
print('s50 pyNuMAD YAML:')
print('  nodes=%d  elems=%d  orient=%d' % (len(nodes), len(els), len(d['elementOrientations'])))
print('  element node-counts:', elens, '(3=tri,4=quad)')
print('  node bbox: x=[%.4f, %.4f]  y=[%.4f, %.4f]  z=[%.4f,%.4f]'
      % (nodes[:,0].min(), nodes[:,0].max(), nodes[:,1].min(), nodes[:,1].max(),
         nodes[:,2].min(), nodes[:,2].max()))
print('  first elem raw:', els[0], ' second:', els[1])
print('  materials order:', [m['name'] for m in d['materials']])
print('  set names:', [s['name'] for s in d['sets']['element']] if isinstance(d['sets'],dict) and 'element' in d['sets'] else [s['name'] for s in d['sets']])

# neighbour s49 .sg bbox (meters, LE-based reference frame the other 49 use)
def sg_bbox(p):
    lines = [l for l in open(p).read().splitlines() if l.strip()]
    for i,l in enumerate(lines):
        t=l.split()
        if len(t)==3 and all(x.lstrip('-').isdigit() for x in t) and int(t[0])>1000:
            nn=int(t[0]); xy=np.array([[float(lines[i+1+k].split()[1]),float(lines[i+1+k].split()[2])] for k in range(nn)])
            return nn, xy
sg49 = ROOT + '/shell51/sg_v201/iea_s49.sg'
if os.path.exists(sg49):
    nn,xy = sg_bbox(sg49)
    print('\ns49.sg (neighbour, reference frame):')
    print('  nodes=%d  x=[%.4f, %.4f]  y=[%.4f, %.4f]' % (nn, xy[:,0].min(), xy[:,0].max(), xy[:,1].min(), xy[:,1].max()))
