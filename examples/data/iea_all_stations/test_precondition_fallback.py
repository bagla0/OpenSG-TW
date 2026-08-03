'''End-to-end proof of the precheck -> condition -> fallback chain on the two stations PreVABS
cannot mesh (s02 eta=0.04, s50 eta=1.0):
  1. condition the XML (shape-preserving)  -> does the verdict improve?
  2. run the pyNuMAD-inspired fallback (OpenSG_io hex_fallback.to_solid_hex) -> 2-D solid YAML
  3. scan the fallback mesh for degeneracy (repair_mesh.diagnose)
  4. homogenize (JAX solid) -> Timo 6x6 diagonal, compared to the spanwise neighbours.'''
import os
import sys
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.expanduser('~/OpenSG-TW-claude')
IO = os.path.join(REPO, 'third_party', 'OpenSG_io')
for q in (REPO, IO, os.path.join(REPO, 'opensg_jax')):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import precheck_prevabs as PC
import condition_prevabs as CD
import repair_mesh as RM

LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']
XMLD = os.path.join(HERE, 'shell51', 'xml')
FBD = os.path.join(HERE, 'shell51', 'fallback_yaml')
os.makedirs(FBD, exist_ok=True)
WINDIO = os.path.join(HERE, 'IEA-22-280-RWT.yaml')

STATIONS = [('s02', 0.04), ('s50', 1.00)]

print('=== step 1+2: condition the flagged XMLs ===')
for tag, eta in STATIONS:
    xml = os.path.join(XMLD, 'iea_%s.xml' % tag)
    try:
        CD.condition(xml, out_dir=os.path.join(XMLD, 'cond'))
    except Exception as e:
        print('  [%s] condition error: %r' % (tag, e))

print('\n=== step 3: pyNuMAD-inspired fallback mesh (to_solid_hex) ===')
from opensg_io import load_blade
from opensg_io.hex_fallback import to_solid_hex
blade = load_blade(WINDIO)
made = {}
for tag, eta in STATIONS:
    out = os.path.join(FBD, 'iea_%s_solid.yaml' % tag)
    ok = False
    for (ms, nr, nw) in [(0.02, 4, 3), (0.03, 3, 2), (0.02, 2, 2)]:
        try:
            info = to_solid_hex(blade, eta, out, mesh_size=ms, nr=nr, nw=nw)
            print('  [%s eta=%.2f] FALLBACK OK  ms=%.2f nr=%d nw=%d  nodes=%d quads=%d webs=%d mats=%s'
                  % (tag, eta, ms, nr, nw, info['n_nodes'], info['n_quads'], info['n_webs'], info['mats']))
            made[tag] = out
            ok = True
            break
        except Exception as e:
            print('  [%s eta=%.2f] ms=%.2f nr=%d nw=%d FAIL: %s' % (tag, eta, ms, nr, nw, repr(e)[:120]))
    if not ok:
        print('  [%s] fallback could not build a valid mesh' % tag)

print('\n=== step 4: scan fallback meshes for degeneracy ===')
for tag, out in made.items():
    s = RM.diagnose(out)
    print('  [%s] nn=%d ne=%d coincident=%d zero=%d slivers=%d' %
          (tag, s['nn'], s['ne'], s['coincident'], s['zero_measure'], s['slivers']))

print('\n=== step 5: homogenize fallback meshes (JAX solid) ===')
import jax
jax.config.update('jax_enable_x64', True)
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml


def d6(p):
    M = np.loadtxt(p)
    return [M[k, k] for k in range(6)]


for tag, out in made.items():
    try:
        K = np.asarray(compute_timo_from_yaml(out, verbose=False))
        print('  [%s] EA=%.4e GA2=%.4e GA3=%.4e GJ=%.4e EI2=%.4e EI3=%.4e'
              % (tag, K[0, 0], K[1, 1], K[2, 2], K[3, 3], K[4, 4], K[5, 5]))
    except Exception:
        traceback.print_exc()

print('\n=== neighbours (existing JAX solids) for trend check ===')
for tag in ('s01', 's03', 's49'):
    p = os.path.join(HERE, 'shell51', 'homo_jax', 'OpenSG_JAX_iea_%s.txt' % tag)
    if os.path.exists(p):
        k = d6(p)
        print('  [%s] EA=%.4e GA2=%.4e GA3=%.4e GJ=%.4e EI2=%.4e EI3=%.4e' % (tag, k[0], k[1], k[2], k[3], k[4], k[5]))
print('DONE')
