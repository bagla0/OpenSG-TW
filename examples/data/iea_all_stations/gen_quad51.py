'''
gen_quad51.py  --  generate ALL 51 IEA-22 cross-sections as 2-D SOLID QUAD meshes using the
pyNuMAD-inspired layered-sweep mesher (OpenSG_io.hex_fallback.to_solid_hex), pure-numpy, no
PreVABS / gmsh / Cubit.  Same geometry source (build_cross_section -> the cs dict) that feeds
the 1-D shell YAML and the PreVABS XML, so the input is identical to the rest of the pipeline.

Output: shell51/pynumad_quad/iea_sNN_solid.yaml   (OpenSG 2-D-solid schema: quad elements,
per-material element sets, [e1,e2,e3] orientations) -- read unchanged by JAX/FEniCS homogenizers.

Uniform eta_i = i/50, i=0..50.  A small param ladder retries degenerate stations (root/tip).
'''
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
IO = os.path.join(REPO, 'third_party', 'OpenSG_io')
for q in (HERE, REPO, IO):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from opensg_io import load_blade
from opensg_io.hex_fallback import to_solid_hex
sys.path.insert(0, os.path.join(HERE, 'shell51'))
from refaxis_shift51 import section_offset_y     # windIO reference-axis (x1) chordwise offset
import precheck_prevabs as PC                     # ms-fast airfoil classifier -> recommended params

WINDIO = os.path.join(HERE, 'IEA-22-280-RWT.yaml')
OUT = os.path.join(HERE, 'shell51', 'pynumad_quad')
os.makedirs(OUT, exist_ok=True)


def shift_to_refaxis(path, eta):
    '''Rigid-translate the just-written mesh from the chord/LE frame to the cross-section
    (0,0) REFERENCE-AXIS origin: subtract section_offset_y(eta) from every node x2 (col0).
    Born-shifted -> the mesh is NEVER in the LE frame. (Do NOT also run refaxis_shift51 here.)'''
    so = section_offset_y(eta)
    lines = open(path).read().splitlines()
    out, in_nodes = [], False
    for ln in lines:
        s = ln.strip()
        if s == 'nodes:':
            in_nodes = True; out.append(ln); continue
        if in_nodes and s and not s.startswith('-'):
            in_nodes = False
        if in_nodes and s.startswith('- ['):
            body = s[s.index('[') + 1:s.rindex(']')]
            t = [float(v) for v in body.replace(',', ' ').split()]
            t[0] -= so
            out.append('- [' + ' '.join('%.8f' % v for v in t) + ']')
        else:
            out.append(ln)
    open(path, 'w').write('\n'.join(out) + '\n')
    return so

LADDER = [(0.02, 4, 3), (0.02, 3, 3), (0.03, 3, 2), (0.02, 2, 2), (0.04, 2, 2),
          (0.015, 6, 2)]   # last rung: near-circular thick ROOT (s00/s01) -- fine skin + more layers


def toks(r):
    if isinstance(r, str):
        return r.split()
    if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
        return r[0].split()
    return [str(x) for x in r]


def scan(path):
    # fast line-parse (no yaml.safe_load) + vectorized degeneracy checks
    nodes, elems, sec = [], [], None
    for ln in open(path):
        s = ln.strip()
        if s.endswith(':') and not s.startswith('-'):
            sec = s.rstrip(':')
            continue
        if s.startswith('- [') and sec in ('nodes', 'elements'):
            body = s[s.index('[') + 1:s.rindex(']')].replace(',', ' ').split()
            if sec == 'nodes':
                nodes.append((float(body[0]), float(body[1])))
            else:
                elems.append([int(x) for x in body[:4]])
    nodes = np.asarray(nodes)
    _, cnt = np.unique(np.round(nodes, 6), axis=0, return_counts=True)
    nc = int((cnt - 1).sum())                      # coincident nodes
    P = nodes[np.asarray(elems) - 1]               # (ne, 4, 2) quad corners
    x, y = P[:, :, 0], P[:, :, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
    zero = int((area < 1e-10).sum())               # zero-area quads
    return len(nodes), len(elems), nc, zero


def main():
    blade = load_blade(WINDIO)
    ok, fail = 0, []
    print('generating 51 pyNuMAD-quad cross-sections -> %s' % OUT, flush=True)
    for i in range(51):
        eta = i / 50.0
        name = 'iea_s%02d' % i
        out = os.path.join(OUT, name + '_solid.yaml')
        t0 = time.time()
        done = None
        # ms-fast precheck classifies the airfoil and RECOMMENDS the mesh params so the
        # near-circular thick root (s00/s01) meshes FIRST TRY instead of walking the ladder.
        xmlp = os.path.join(HERE, 'shell51', 'xml', name + '.xml')
        rec = None
        if os.path.exists(xmlp):
            qp = PC.precheck(xmlp).get('quad_params')
            if qp:
                rec = (qp['mesh_size'], qp['nr'], qp['nw'])
        ladder = ([rec] + [r for r in LADDER if r != rec]) if rec else LADDER
        for (ms, nr, nw) in ladder:
            try:
                info = to_solid_hex(blade, eta, out, mesh_size=ms, nr=nr, nw=nw)
                done = (ms, nr, nw, info)
                break
            except Exception as e:
                last = repr(e)[:90]
        if done:
            ms, nr, nw, info = done
            so = shift_to_refaxis(out, eta)       # -> cross-section (0,0) reference-axis origin
            nn, ne, nc, zero = scan(out)
            flag = '' if (nc == 0 and zero == 0) else '  <-- degenerate!'
            print('[%s] eta=%.2f  ms=%.2f nr=%d nw=%d  nodes=%d quads=%d webs=%d  x1-shift=%+.3f  coincident=%d zero=%d  [%.1fs]%s'
                  % (name, eta, ms, nr, nw, info['n_nodes'], info['n_quads'], info['n_webs'], so, nc, zero, time.time() - t0, flag),
                  flush=True)
            ok += 1
        else:
            print('[%s] eta=%.2f  FAILED all ladder rungs: %s' % (name, eta, last), flush=True)
            fail.append(name)
    print('\n%d/51 built via pyNuMAD-quad.  failed: %s' % (ok, fail if fail else 'NONE'), flush=True)


if __name__ == '__main__':
    main()
