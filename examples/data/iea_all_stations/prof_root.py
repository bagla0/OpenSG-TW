'''Profile the pyNuMAD-quad build for the slow root (s00, 9.5s) vs a fast station (s02, 2.3s)
to locate the bottleneck in to_solid_hex / hex_between_sections.'''
import os
import sys
import cProfile
import pstats
import io
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
IO = os.path.join(REPO, 'third_party', 'OpenSG_io')
for q in (HERE, REPO, IO):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from opensg_io import load_blade
from opensg_io.hex_fallback import to_solid_hex

blade = load_blade(os.path.join(HERE, 'IEA-22-280-RWT.yaml'))
OUT = os.path.join(HERE, 'shell51', 'pynumad_quad')


def prof(tag, eta, ms, nr, nw):
    out = os.path.join(OUT, '_prof_%s.yaml' % tag)
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    to_solid_hex(blade, eta, out, mesh_size=ms, nr=nr, nw=nw)
    pr.disable()
    dt = time.time() - t0
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(15)
    print('\n================= %s eta=%.2f ms=%.3f nr=%d nw=%d  TOTAL %.2fs =================' % (tag, eta, ms, nr, nw, dt))
    # print only the function rows (skip the header noise)
    for ln in s.getvalue().splitlines():
        if 'opensg_io' in ln or 'hex_loft' in ln or 'section_offset' in ln or 'ncalls' in ln or 'numpy' in ln:
            print(ln)
    try:
        os.remove(out)
    except OSError:
        pass


prof('s00_root', 0.0, 0.015, 6, 2)
prof('s02_fast', 0.04, 0.02, 4, 3)
