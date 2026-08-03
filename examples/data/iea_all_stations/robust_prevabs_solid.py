#!/usr/bin/env python
r'''
robust_prevabs_solid.py  --  robust windIO -> 2-D solid cross-section, with automatic
preconditioning and a pyNuMAD-inspired fallback.

Decision path (exactly the requested behaviour):

    emit XML  ->  precheck_prevabs
        |
        +-- CLEAN / WARN  ------> run PreVABS ---success--> 2-D solid YAML          [method=prevabs]
        |                                    \--fail--\
        +-- FATAL_FIXABLE -> condition_prevabs (shape-preserving) -> run PreVABS
        |                                    |--success--> 2-D solid YAML   [method=prevabs-conditioned]
        |                                    \--fail-----\
        +-- UNFIXABLE  ------------------------------------> "PreVABS cannot handle it"
                                                             -> pyNuMAD-inspired FALLBACK
                                                                (OpenSG_io hex_fallback.to_solid_hex,
                                                                 pure-numpy layered-quad, no PreVABS)
                                                             -> 2-D solid YAML            [method=fallback]

Every branch yields the SAME OpenSG 2-D-solid YAML schema, read unchanged by the JAX/FEniCS
homogenizers.  The fallback is only used when PreVABS provably cannot mesh the station
(the run is actually attempted first), so accuracy is never silently traded away.

USAGE
    python robust_prevabs_solid.py <windio.yaml> <name> <eta> <xmldir> <out.yaml> [--prevabs BIN]
e.g. python robust_prevabs_solid.py IEA-22-280-RWT.yaml iea_s02 0.04 shell51/xml shell51/2d_yaml/iea_s02_solid.yaml

Importable:
    from robust_prevabs_solid import build_solid_robust
    info = build_solid_robust(blade, eta, name, xmldir, out_yaml, prevabs_bin)
'''
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
IO = os.path.join(REPO, 'third_party', 'OpenSG_io')
for q in (HERE, REPO, IO):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import precheck_prevabs as PC
import condition_prevabs as CD

DEFAULT_PREVABS = os.path.expanduser('~/prevabs-fork/build/prevabs')


def _try_prevabs(xml, out_yaml, prevabs_bin, timeout=180):
    from opensg_io.prevabs_xml import to_solid
    to_solid(xml, out_yaml, prevabs_bin, timeout=timeout)     # raises on any failure
    if not (os.path.exists(out_yaml) and os.path.getsize(out_yaml) > 0):
        raise RuntimeError('PreVABS produced no YAML')
    return True


def build_solid_robust(blade, eta, name, xmldir, out_yaml, prevabs_bin=DEFAULT_PREVABS,
                       mesh_size=0.01, nr=4, nw=3, verbose=True):
    xml = os.path.join(xmldir, name + '.xml')
    if not os.path.exists(xml):
        raise FileNotFoundError('XML not found (emit it with emit_prevabs first): %s' % xml)
    rep = PC.precheck(xml)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    log('[%s] precheck verdict: %s' % (name, rep['verdict']))

    # attempt 1: PreVABS as-is unless the precheck is already fatal
    if rep['verdict'] in ('CLEAN', 'WARN_LIKELY_OK'):
        try:
            _try_prevabs(xml, out_yaml, prevabs_bin)
            log('[%s] -> meshed by PreVABS (as generated)' % name)
            return dict(name=name, method='prevabs', out=out_yaml, verdict=rep['verdict'])
        except Exception as e:
            log('[%s] PreVABS failed as-is (%s); trying preprocessing' % (name, repr(e)[:80]))

    # attempt 2: shape-preserving preprocessing, then PreVABS
    try:
        info = CD.condition(xml, out_dir=os.path.join(xmldir, 'cond'), verbose=False)
        cxml = info['new_xml']
        log('[%s] conditioned (%d fix(es)): %s' % (name, len(info['fixes']), '; '.join(info['fixes'])[:160]))
        if cxml:
            _try_prevabs(cxml, out_yaml, prevabs_bin)
            log('[%s] -> meshed by PreVABS after preprocessing' % name)
            return dict(name=name, method='prevabs-conditioned', out=out_yaml, verdict=rep['verdict'])
    except Exception as e:
        log('[%s] PreVABS still failed after preprocessing (%s)' % (name, repr(e)[:80]))

    # attempt 3: PreVABS cannot handle it -> pyNuMAD-inspired fallback
    log('[%s] PreVABS cannot handle this cross-section -> pyNuMAD-inspired fallback (to_solid_hex)' % name)
    from opensg_io.hex_fallback import to_solid_hex
    last = None
    for (ms, r_, w_) in [(mesh_size, nr, nw), (max(mesh_size, 0.02), max(nr - 1, 2), max(nw - 1, 2)),
                         (max(mesh_size, 0.03), 2, 2)]:
        try:
            hi = to_solid_hex(blade, eta, out_yaml, mesh_size=ms, nr=r_, nw=w_)
            log('[%s] -> FALLBACK mesh built: nodes=%d quads=%d webs=%d' % (name, hi['n_nodes'], hi['n_quads'], hi['n_webs']))
            return dict(name=name, method='fallback', out=out_yaml, verdict=rep['verdict'], **hi)
        except Exception as e:
            last = e
    raise RuntimeError('[%s] neither PreVABS nor the fallback could mesh this station: %r' % (name, last))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Robust windIO -> 2-D solid cross-section (precheck+condition+PreVABS+fallback).')
    ap.add_argument('windio')
    ap.add_argument('name')
    ap.add_argument('eta', type=float)
    ap.add_argument('xmldir')
    ap.add_argument('out_yaml')
    ap.add_argument('--prevabs', default=DEFAULT_PREVABS)
    ap.add_argument('--mesh-size', type=float, default=0.01)
    a = ap.parse_args(argv)
    from opensg_io import load_blade
    blade = load_blade(a.windio)
    info = build_solid_robust(blade, a.eta, a.name, a.xmldir, a.out_yaml, a.prevabs, mesh_size=a.mesh_size)
    print('RESULT: %s' % info)
    return 0


if __name__ == '__main__':
    sys.exit(main())
