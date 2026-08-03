'''Confirm the _fold_nodes vectorization: s00/s01/s02 build time + node/quad counts
(must MATCH the pre-optimization mesh: s00/s01 = 2087 nodes/1688 quads, s02 = 1350/1061)
+ degeneracy-clean. Then homogenize s00 to confirm the Timo 6x6 is unchanged.'''
import os
import sys
import time
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
IO = os.path.join(REPO, 'third_party', 'OpenSG_io')
for q in (HERE, REPO, IO, os.path.join(REPO, 'opensg_jax')):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from opensg_io import load_blade
from opensg_io.hex_fallback import to_solid_hex

blade = load_blade(os.path.join(HERE, 'IEA-22-280-RWT.yaml'))
OUT = os.path.join(HERE, 'shell51', 'pynumad_quad')
KNOWN = {'s00': (0.0, 0.015, 6, 2), 's01': (0.02, 0.015, 6, 2), 's02': (0.04, 0.02, 4, 3)}

print('=== build time + counts (vectorized _fold_nodes) ===')
for tag, (eta, ms, nr, nw) in KNOWN.items():
    out = os.path.join(OUT, 'iea_%s_solid.yaml' % tag)
    t0 = time.time()
    info = to_solid_hex(blade, eta, out, mesh_size=ms, nr=nr, nw=nw)
    dt = time.time() - t0
    print('  %s: %.2fs  nodes=%d quads=%d webs=%d' % (tag, dt, info['n_nodes'], info['n_quads'], info['n_webs']))
