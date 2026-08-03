'''refaxis_shift51.py -- rigid-translate iea_sNN yamls from the LE origin to the windIO
REFERENCE-AXIS origin (x1): subtract section_offset_y(eta_i), eta_i=i/50, from every node x2 (col0).

Works for BOTH the 1-D shell yaml and the 2-D solid yaml (node line '- [x2 x3 x?]').  Only node
coordinates change (elements/orientations/sets/materials untouched) so EA/GA/GJ are unchanged;
EI2/EI3 move onto the reference axis.  Per-file sidecar marker '<yaml>.refaxis' makes it idempotent:
a file already shifted (matching so) is skipped; delete the marker to force a re-shift (e.g. after a
PreVABS regeneration writes a fresh LE mesh).

    python refaxis_shift51.py --dir 1d_yaml --glob '*_shell.yaml'
    python refaxis_shift51.py --dir 2d_yaml --glob '*_solid.yaml'
'''
import argparse
import glob
import os
import re

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                     # iea_all_stations
WINDIO = os.path.join(BASE, 'IEA-22-280-RWT.yaml')

_d = yaml.safe_load(open(WINDIO))
_so = _d['components']['blade']['outer_shape']['section_offset_y']
SO_G, SO_V = np.array(_so['grid'], float), np.array(_so['values'], float)


def section_offset_y(eta):
    return float(np.interp(eta, SO_G, SO_V))


def station_eta(name):
    m = re.search(r'iea_s(\d+)', name)
    if not m:
        return None
    return int(m.group(1)) / 50.0


def shift_file(f):
    name = os.path.basename(f)
    eta = station_eta(name)
    if eta is None:
        return None
    so = section_offset_y(eta)
    marker = f + '.refaxis'
    if os.path.exists(marker):
        return ('skip', so)
    lines = open(f).read().splitlines()
    out, in_nodes = [], False
    xs = []
    for ln in lines:
        s = ln.strip()
        if s == 'nodes:':
            in_nodes = True; out.append(ln); continue
        if in_nodes and s and not s.startswith('-'):    # reached next key (elements: etc.)
            in_nodes = False
        if in_nodes and s.startswith('- ['):
            body = s[s.index('[') + 1:s.rindex(']')]
            t = [float(v) for v in body.replace(',', ' ').split()]
            t[0] -= so; xs.append(t[0])
            out.append('- [' + ' '.join('%.8f' % v for v in t) + ']')
        else:
            out.append(ln)
    with open(f, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    open(marker, 'w').write('so=%.10f eta=%.6f\n' % (so, eta))
    return ('shift', so, min(xs), max(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--glob', default='*.yaml')
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(HERE, a.dir)
    files = sorted(glob.glob(os.path.join(d, a.glob)))
    ns = nk = 0
    for f in files:
        r = shift_file(f)
        if r is None:
            continue
        if r[0] == 'skip':
            nk += 1
        else:
            ns += 1
            print('%-28s so=%+.4f  x2 in [%.3f, %.3f]' %
                  (os.path.basename(f), r[1], r[2], r[3]))
    print('\nshifted %d, skipped(already) %d  in %s' % (ns, nk, d))


if __name__ == '__main__':
    main()
