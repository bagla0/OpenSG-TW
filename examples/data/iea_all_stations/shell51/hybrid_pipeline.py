'''Hybrid 2-D cross-section pipeline (the user's strategy):
  DEFAULT cross-sections  -> PreVABS TRIANGULAR refined mesh (validated <1% vs VABS); these are the
                             49 solids already in 2d_yaml/ (the current fork binary can no longer
                             regenerate them -- it is broken in both modes -- so we reuse the
                             validated ones + their homogenized 6x6 in homo_jax_prevabs_bak/).
  PRECHECK-FLAGGED (s02,s50, the sections PreVABS never meshed) -> REFINED QUAD (to_solid_hex nr=8).
Assembles 2d_hybrid/, homogenizes (JAX solid), emits .out, regenerates the %-error plot.'''
import os
import sys
import glob
import shutil
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.expanduser('~/OpenSG-TW-claude')
IO = os.path.join(REPO, 'third_party', 'OpenSG_io')
for q in (HERE, REPO, IO, os.path.join(REPO, 'opensg_jax')):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from opensg_io import load_blade
from opensg_io.hex_fallback import to_solid_hex
from refaxis_shift51 import section_offset_y
import jax
jax.config.update('jax_enable_x64', True)
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml

WINDIO = os.path.join(BASE, 'IEA-22-280-RWT.yaml')
BAK = os.path.join(HERE, 'homo_jax_prevabs_bak')
JX = os.path.join(HERE, 'homo_jax'); os.makedirs(JX, exist_ok=True)
HYB = os.path.join(HERE, '2d_hybrid'); os.makedirs(HYB, exist_ok=True)
TRI = os.path.join(HERE, '2d_yaml')
PY = sys.executable


def shift(path, eta):
    so = section_offset_y(eta)
    lines = open(path).read().splitlines()
    out, inn = [], False
    for ln in lines:
        s = ln.strip()
        if s == 'nodes:':
            inn = True; out.append(ln); continue
        if inn and s and not s.startswith('-'):
            inn = False
        if inn and s.startswith('- ['):
            b = [float(v) for v in s[s.index('[') + 1:s.rindex(']')].replace(',', ' ').split()]
            b[0] -= so
            out.append('- [' + ' '.join('%.8f' % v for v in b) + ']')
        else:
            out.append(ln)
    open(path, 'w').write('\n'.join(out) + '\n')
    return so


# 1. restore the validated PreVABS-TRI solid 6x6 into homo_jax
n = 0
for f in glob.glob(os.path.join(BAK, '*.txt')):
    shutil.copy(f, JX); n += 1
print('restored %d PreVABS-tri solid 6x6 from backup' % n)

# 2. assemble 2d_hybrid with the tri solids; find the flagged (no-tri) stations
tri_tags = set()
for f in sorted(glob.glob(os.path.join(TRI, '*_solid.yaml'))):
    if 't1only' in f:
        continue
    tag = os.path.basename(f).replace('_solid.yaml', '')
    shutil.copy(f, os.path.join(HYB, tag + '_solid.yaml'))
    tri_tags.add(tag)
all_tags = ['iea_s%02d' % i for i in range(51)]
flagged = [t for t in all_tags if t not in tri_tags]
print('DEFAULT (PreVABS tri): %d stations   FLAGGED (refined quad): %s' % (len(tri_tags), flagged))

# 3. refined quad (nr=8) for the flagged sections -> hybrid mesh + homogenize
blade = load_blade(WINDIO)
t0 = time.time()
for tag in flagged:
    i = int(tag.split('s')[-1]); eta = i / 50.0
    out = os.path.join(HYB, tag + '_solid.yaml')
    to_solid_hex(blade, eta, out, mesh_size=0.02, nr=8, nw=3)   # REFINED quad
    so = shift(out, eta)
    K = np.asarray(compute_timo_from_yaml(out, verbose=False))
    np.savetxt(os.path.join(JX, 'OpenSG_JAX_%s.txt' % tag), K)
    print('  refined-quad %s (nr=8) x1-shift=%+.3f  EA=%.3e GA3=%.3e' % (tag, so, K[0, 0], K[2, 2]))

# 4. any tri without a restored 6x6 -> homogenize now
for tag in sorted(tri_tags):
    if not os.path.exists(os.path.join(JX, 'OpenSG_JAX_%s.txt' % tag)):
        K = np.asarray(compute_timo_from_yaml(os.path.join(HYB, tag + '_solid.yaml'), verbose=False))
        np.savetxt(os.path.join(JX, 'OpenSG_JAX_%s.txt' % tag), K)
        print('  (re-homogenized missing tri %s)' % tag)
print('flagged-quad + missing-tri homogenization: %.1f s' % (time.time() - t0))

# 5. emit .out (hybrid solid) + RM shell, then the %-error plot
te = time.time()
subprocess.run([PY, os.path.join(HERE, 'emit_full_out51.py'), '--source', 'jax',
                '--ydir', '2d_hybrid', '--outsuffix', 'OpenSG_Hybrid_Solid'], check=False)
subprocess.run([PY, os.path.join(HERE, 'emit_full_out51.py'), '--source', 'rm'], check=False)
print('.out generation: %.1f s' % (time.time() - te))
subprocess.run([PY, os.path.join(HERE, 'plot_pcterr.py')], check=False)
print('HYBRID_DONE')
