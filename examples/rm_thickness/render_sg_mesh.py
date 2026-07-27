"""render_sg_mesh.py -- simple PyVista renders of the ACTUAL 1-D structure-gene
line-element meshes read back from the sg_yaml/ input files: line elements coloured
by ply (material + angle), FE nodes as points.

Output: figures/sg_mesh_ex1.png, figures/sg_mesh_ex2.png.
"""
import os

import numpy as np
import yaml
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
YDIR = os.path.join(HERE, 'sg_yaml')
OUT = os.path.join(HERE, 'figures')

PALETTE = ['royalblue', 'darkorange', 'seagreen', 'firebrick', 'purple',
           'goldenrod', 'teal', 'hotpink']
NICE = {'pagano': 'graphite/epoxy', 'as4': 'AS4-type', 'face': 'face',
        'core': 'soft core', 'mr_lam': 'graphite/epoxy', 'mr_face': 'face',
        'mr_core': 'soft core'}


def load(name):
    d = yaml.safe_load(open(os.path.join(YDIR, name + '.yaml'), encoding='utf-8'))
    nodes = np.array(d['nodes'], float)
    elems = np.array(d['elements'], int) - 1
    lay = d['sections'][0]['layup']
    eply = {}
    for k, s in enumerate(d['sets']['element']):
        for lab in s['labels']:
            eply[lab - 1] = k
    return nodes, elems, lay, eply


def figure(names, labels, fname):
    pl = pv.Plotter(off_screen=True, window_size=(1250, 850))
    pl.set_background('white')
    hmax = max(load(n)[0][:, 2].max() for n in names)
    groups = {}                     # (material, angle) -> colour
    for i, (name, lab) in enumerate(zip(names, labels)):
        nodes, elems, lay, eply = load(name)
        x = i * 0.45 * hmax
        for e, (n1, n2) in enumerate(elems):
            mat, t, ang = lay[eply[e]]
            key = (mat, float(ang))
            if key not in groups:
                groups[key] = PALETTE[len(groups) % len(PALETTE)]
            seg = pv.Line((x, 0, nodes[n1, 2]), (x, 0, nodes[n2, 2]))
            pl.add_mesh(seg, color=groups[key], line_width=7)
        pl.add_point_labels([(x, 0, -0.05 * hmax)], [lab], font_size=24,
                            shape=None, always_visible=True, show_points=False)
    pl.add_legend([(f"{NICE.get(m, m)}, {ang:g}$^\\circ$", c)
                   for (m, ang), c in groups.items()],
                  bcolor='white', border=True, size=(0.30, 0.035 * len(groups) + 0.03),
                  loc='lower right')
    pl.camera_position = 'xz'
    pl.enable_parallel_projection()
    pl.camera.zoom(1.05)
    path = os.path.join(OUT, fname)
    pl.screenshot(path)
    print('wrote', path)


if __name__ == '__main__':
    figure(['ex1_pagano_090_0', 'ex1_as4_sym', 'ex1_sandwich'],
           ['[0/90/0]', '[0/90/90/0]', 'sandwich'], 'sg_mesh_ex1.png')
    figure(['ex2_mr_sym', 'ex2_mr_asym', 'ex2_mr_sandwich'],
           ['[0/90/0]', '[0/90]', 'sandwich'], 'sg_mesh_ex2.png')
