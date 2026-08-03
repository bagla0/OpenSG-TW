'''Diagnose why the pyNuMAD-quad (hex loft) fails at the ROOT (s00 eta=0, s01 eta=0.02):
inspect chord, webs (positions/length), segments, and where hex_between_sections inverts cells.'''
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
IO = os.path.join(REPO, 'third_party', 'OpenSG_io')
for q in (HERE, REPO, IO):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from opensg_io import load_blade
from opensg_io.converter import build_cross_section
from opensg_io.hex_loft import hex_between_sections

blade = load_blade(os.path.join(HERE, 'IEA-22-280-RWT.yaml'))

for eta in (0.0, 0.02, 0.04, 0.10):
    print('\n===== eta=%.2f =====' % eta)
    try:
        cs = build_cross_section(blade, eta, mesh_size=0.02)
    except Exception as e:
        print('  build_cross_section FAILED: %r' % e)
        continue
    nodes = np.asarray(cs['nodes'])
    print('  chord=%.4f  n_nodes=%d  n_segments=%d  n_webs=%d  n_laminates=%d'
          % (cs['chord'], len(nodes), len(cs['segments']), len(cs['webs']), len(cs['laminates'])))
    for wi, w in enumerate(cs['webs']):
        a, b = w['a'], w['b']
        Pa, Pb = nodes[a], nodes[b]
        L = np.linalg.norm(np.asarray(Pa) - np.asarray(Pb))
        print('   web %d: a=%d b=%d  len=%.4f  lam=%s  Pa=(%.3f,%.3f) Pb=(%.3f,%.3f)'
              % (wi, a, b, L, w['lam'], Pa[0], Pa[1], Pb[0], Pb[1]))
    # try the loft
    for (nr, nw) in [(4, 3), (2, 2)]:
        try:
            res = hex_between_sections(cs, cs, 0.0, 1.0, nr=nr, nsp=1, nw=nw, mesh_size=0.02)
            print('   loft nr=%d nw=%d -> n_still_inverted=%d  n_hexes=%s'
                  % (nr, nw, res.get('n_still_inverted', -1), len(res.get('hexes', []))))
        except Exception as e:
            print('   loft nr=%d nw=%d FAILED: %r' % (nr, nw, repr(e)[:100]))
