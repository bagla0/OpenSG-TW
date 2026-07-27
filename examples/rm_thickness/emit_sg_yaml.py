"""emit_sg_yaml.py -- write the OpenSG 1-D structure-gene YAML input for every
benchmark laminate of Examples 1 and 2, and verify the round trip.

The schema mirrors the OpenSG shell SG yaml (cf. iea_s10_shell.yaml): ``nodes`` (the
through-thickness FE nodes actually used by the solver), ``elements`` (2-node line
connectivity between ply-element endpoints), ``sets.element`` (one set per ply),
``sections`` (type: sg1d, layup = [material, thickness, angle] outer->inner) and
``materials`` (E/G/nu triplets + density).  ``load_sg_yaml`` reads a file back into
the (thick, angles, mats, matdb) tuple the solver consumes; the __main__ block writes
every file and asserts that the yaml-loaded laminate reproduces the inline-defined
A6 and G_MSG to round-off.
"""
import os

import numpy as np
import yaml

from jaxcfg import jnp                      # noqa: F401  (x64 first)
import sg_plate as SG
from materials import MATDB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'sg_yaml')

# ---- Example 1 laminates (cylindrical bending, total h = 1 m; S varies via L) ----
# ---- Example 2 laminates (bidirectional plate, H = a/aspect; a/H=4 shown) -------
try:
    from navier_plate.navier_models import MATDB_MR
except ImportError:  # run from inside the folder
    import sys
    sys.path.insert(0, os.path.join(HERE, 'navier_plate'))
    from navier_models import MATDB_MR

ALLDB = {**MATDB, **MATDB_MR}

CASES = {
    # Example 1 (h = 1)
    'ex1_pagano_090_0':  ([1 / 3] * 3, [0., 90., 0.], ['pagano'] * 3),
    'ex1_pagano_90090':  ([1 / 3] * 3, [90., 0., 90.], ['pagano'] * 3),
    'ex1_as4_sym':       ([0.25] * 4, [0., 90., 90., 0.], ['as4'] * 4),
    'ex1_as4_asym':      ([0.25] * 4, [0., 90., 0., 90.], ['as4'] * 4),
    'ex1_sandwich':      ([0.1, 0.8, 0.1], [0.] * 3, ['face', 'core', 'face']),
    'ex1_angle_0450':    ([1 / 3] * 3, [0., 45., 0.], ['pagano'] * 3),
    # Example 2 (H = 0.25 m shown; the a/H = 100 runs scale the same stack by 0.04)
    'ex2_mr_sym':        ([0.25 / 3] * 3, [0., 90., 0.], ['mr_lam'] * 3),
    'ex2_mr_asym':       ([0.125] * 2, [0., 90.], ['mr_lam'] * 2),
    'ex2_mr_sandwich':   ([0.025, 0.2, 0.025], [0.] * 3,
                          ['mr_face', 'mr_core', 'mr_face']),
}
NPL = {'ex1_sandwich': 8, 'ex2_mr_sandwich': 8}     # elements per ply (default 6)


def emit(name, thick, angles, mats, n_per_layer, p=3):
    """Write one SG yaml carrying the ACTUAL solver mesh."""
    m = SG.sg_mesh(np.asarray(thick, float), n_per_layer, p, 0.0)
    nodes = [[0.0, 0.0, float(x)] for x in m['node_x']]
    elements = [[int(p * e + 1), int(p * e + p + 1)] for e in range(m['n_elem'])]
    sets = []
    for k in range(len(thick)):
        labs = [int(e + 1) for e in range(m['n_elem']) if m['elem_layer'][e] == k]
        sets.append({'name': f'ply_{k}', 'labels': labs})
    matdb = {}
    for mt in dict.fromkeys(mats):
        d = ALLDB[mt]
        matdb[mt] = {'density': float(d.get('rho', 1.0)),
                     'elastic': {'E': [float(x) for x in d['E']],
                                 'G': [float(x) for x in d['G']],
                                 'nu': [float(x) for x in d['nu']]}}
    doc = {
        'model': 'OpenSG 1-D structure gene (through-thickness)',
        'element_order': p,
        'elements_per_ply': n_per_layer,
        'nodes': nodes,
        'elements': elements,
        'sets': {'element': sets},
        'sections': [{'type': 'sg1d',
                      'layup': [[mats[k], float(thick[k]), float(angles[k])]
                                for k in range(len(thick))]}],
        'materials': matdb,
    }
    path = os.path.join(OUT, name + '.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
    return path


def load_sg_yaml(path):
    """Read an SG yaml back into (thick, angles, mats, matdb, n_per_layer, p)."""
    d = yaml.safe_load(open(path, encoding='utf-8'))
    lay = d['sections'][0]['layup']
    mats = [r[0] for r in lay]
    thick = [float(r[1]) for r in lay]
    angles = [float(r[2]) for r in lay]
    matdb = {}
    for name, mm in d['materials'].items():
        e = mm['elastic']
        matdb[name] = {'E': e['E'], 'G': e['G'], 'nu': e['nu'],
                       'rho': mm.get('density', 1.0)}
    return thick, angles, mats, matdb, int(d['elements_per_ply']), int(d['element_order'])


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, (thick, angles, mats) in CASES.items():
        npl = NPL.get(name, 6)
        path = emit(name, thick, angles, mats, npl)
        # round trip: yaml-loaded laminate must reproduce the inline solve exactly
        t2, a2, m2, db2, npl2, p2 = load_sg_yaml(path)
        sg_a = SG.build(thick, angles, mats, ALLDB, n_per_layer=npl, elem_order=3)
        sg_b = SG.build(t2, a2, m2, db2, n_per_layer=npl2, elem_order=p2)
        dA = float(np.max(np.abs(np.asarray(sg_a['A6']) - np.asarray(sg_b['A6']))))
        dG = float(np.max(np.abs(np.asarray(sg_a['G_msg']) - np.asarray(sg_b['G_msg']))))
        rel = dA / float(np.max(np.abs(np.asarray(sg_a['A6']))))
        print(f"{name:<18} nodes {len(yaml.safe_load(open(path))['nodes']):>3}  "
              f"roundtrip |dA|/|A| {rel:.1e}  |dG| {dG:.1e}")
    print(f"wrote {len(CASES)} yamls -> {OUT}")
