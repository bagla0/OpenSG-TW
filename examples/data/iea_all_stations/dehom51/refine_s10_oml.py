'''refine_s10_oml.py -- subdivide every element of the OML ring N times (chord
midpoints; layup set membership and orientation inherited) so the RAW dehom
shear-flow recovery converges without any post-processing.  Writes
shell51/1d_yaml_oml/iea_s10_shell_ref<N>.yaml and prints the Timoshenko diag
of parent vs refined (must be ~unchanged) plus the raw jump statistics of the
two contour-derivative strain rows (must SHRINK ~1/N if the oscillation is the
linear-interpolation artifact, and persist if it is something deeper).
'''
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, '..'))
XS = next(c for c in [os.path.expanduser('~/OpenSG-TW-claude/examples/TW-paper/xsec_paper'),
                      r'Y:\OpenSG-TW-claude\examples\TW-paper\xsec_paper'] if os.path.isdir(c))
REPO = os.path.abspath(os.path.join(XS, '..', '..', '..'))
for q in (XS, REPO, os.path.join(REPO, 'mitc_rm_segment')):
    sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import jax

jax.config.update('jax_enable_x64', True)
import dehom_rm
from dehom_rm import _macro_fields, _rm_shell_strain

SRC = os.path.join(IEA, 'shell51', '1d_yaml_oml', 'iea_s10_shell.yaml')
FF = np.loadtxt(os.path.join(HERE, 'beamdyn', 'ff51_rmc_reform.dat'))[10, 1:]
NS = [2, 4]


def rows(seq):
    out = []
    for r in seq:
        if isinstance(r, str):
            out.append([float(x) for x in r.split()])
        elif isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
            out.append([float(x) for x in r[0].split()])
        else:
            out.append([float(x) for x in r])
    return np.array(out)


def refine(src, n, dst):
    d = yaml.safe_load(open(src))
    nd = rows(d['nodes'])
    cells = rows(d['elements']).astype(int)
    one = cells.min() == 1
    if one:
        cells = cells - 1
    ori = d['elementOrientations']
    new_nodes = [list(map(float, r)) for r in nd]
    new_cells = []
    new_ori = []
    child_of = {}
    for e, (a, b) in enumerate(cells):
        pa, pb = nd[a][:2], nd[b][:2]
        ids = [a]
        for k in range(1, n):
            t = k / n
            p = [(1 - t) * pa[0] + t * pb[0], (1 - t) * pa[1] + t * pb[1], 0.0]
            new_nodes.append(p)
            ids.append(len(new_nodes) - 1)
        ids.append(b)
        kids = []
        for k in range(n):
            new_cells.append([ids[k] + 1, ids[k + 1] + 1])
            new_ori.append(ori[e])
            kids.append(len(new_cells))       # 1-indexed child label
        child_of[e + 1] = kids                # parent label (1-indexed) -> children
    d['nodes'] = [[float(p[0]), float(p[1]), float(p[2] if len(p) > 2 else 0.0)]
                  for p in new_nodes]
    d['elements'] = [[int(c[0]), int(c[1])] for c in new_cells]
    d['elementOrientations'] = new_ori
    for g in d['sets']['element']:
        lab = []
        for L in g['labels']:
            lab.extend(child_of[int(L)])
        g['labels'] = lab
    with open(dst, 'w') as f:
        yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False, width=1000)
    return dst


def jumpstats(shell_yaml, tag):
    B = dehom_rm.build_rm_bundle(shell_yaml)
    st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
    rc = np.asarray(B['red_cells'])
    n_el = rc.shape[0]
    v2 = np.zeros(n_el)
    v5 = np.zeros(n_el)
    for e in range(n_el):
        s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        v2[e] = float(s6[2])
        v5[e] = float(s6[5])
    d = np.diag(np.asarray(B['Timo']))
    print('%-8s elems %4d  Timo diag %s' % (tag, n_el,
          np.array2string(d, precision=3, formatter={'float_kind': lambda x: '%.3e' % x})))
    for nm, v in (('2eps12', v2), ('2k12  ', v5)):
        dj = np.abs(np.diff(v))
        print('   %s  mean|jump| %.3e  mean|val| %.3e  ratio %.2f'
              % (nm, dj.mean(), np.abs(v).mean(), dj.mean() / (np.abs(v).mean() + 1e-30)))
    return B


jumpstats(SRC, 'ref1')
for n in NS:
    dst = SRC.replace('.yaml', '_ref%d.yaml' % n)
    refine(SRC, n, dst)
    jumpstats(dst, 'ref%d' % n)
