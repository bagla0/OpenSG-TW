'''Wider param sweep for the two root stations (eta=0.0, 0.02): can ANY (mesh_size,nr,nw)
give the hex loft 0 inverted cells? If yes, add it to gen_quad51's ladder.'''
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
GRID = [(ms, nr, nw) for ms in (0.015, 0.02, 0.03, 0.04, 0.05)
        for nr in (2, 3, 4, 6) for nw in (2, 3, 4, 5)]

for eta in (0.0, 0.02):
    print('\n===== eta=%.2f =====' % eta)
    best = None
    for (ms, nr, nw) in GRID:
        try:
            cs = build_cross_section(blade, eta, mesh_size=ms)
            res = hex_between_sections(cs, cs, 0.0, 1.0, nr=nr, nsp=1, nw=nw, mesh_size=ms)
            inv = res.get('n_still_inverted', -1)
            if inv == 0:
                print('  CLEAN: ms=%.3f nr=%d nw=%d  hexes=%d' % (ms, nr, nw, len(res['hexes'])))
                best = (ms, nr, nw)
                break
            if best is None or inv < best[0]:
                pass
        except Exception:
            pass
    if best is None:
        # report the minimum inversions achievable
        mn = 999
        for (ms, nr, nw) in GRID:
            try:
                cs = build_cross_section(blade, eta, mesh_size=ms)
                res = hex_between_sections(cs, cs, 0.0, 1.0, nr=nr, nsp=1, nw=nw, mesh_size=ms)
                inv = res.get('n_still_inverted', -1)
                if 0 <= inv < mn:
                    mn = inv; arg = (ms, nr, nw)
            except Exception:
                pass
        print('  NO clean combo; min inverted=%d at %s' % (mn, arg if mn < 999 else 'n/a'))
