'''Generate the explanatory figures for the PreVABS thin/thick meshing report.
 fig1_meshes.png    -- the actual pyNuMAD-quad meshes that succeed at the ROOT (s00) and TIP (s50)
 fig2_pipeline.png  -- PreVABS (Clipper2 offset) vs pyNuMAD (per-vertex sweep) pipelines side by side
 fig3_mechanism.png -- WHY: Clipper2 polygon-inflate collapses on a thin wall vs the sweep never empties
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
QUAD = os.path.join(HERE, 'shell51', 'pynumad_quad')
OUT = os.path.join(HERE, 'prevabs_report')
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------- mesh loader + render
def load_mesh(path):
    nodes, elems, sec = [], [], None
    setmap, curset = {}, None
    for ln in open(path):
        s = ln.strip()
        if s.endswith(':') and not s.startswith('-'):
            sec = s.rstrip(':'); continue
        if sec == 'nodes' and s.startswith('- ['):
            b = s[s.index('[') + 1:s.rindex(']')].replace(',', ' ').split()
            nodes.append((float(b[0]), float(b[1])))
        elif sec == 'elements' and s.startswith('- ['):
            b = s[s.index('[') + 1:s.rindex(']')].replace(',', ' ').split()
            elems.append([int(x) for x in b[:4]])
        elif sec == 'sets':
            if s.startswith('- name:'):
                curset = s.split('name:')[1].strip()
            elif s.startswith('labels:'):
                lab = s[s.index('[') + 1:s.rindex(']')].replace(',', ' ').split()
                setmap[curset] = [int(x) for x in lab]
    nodes = np.asarray(nodes)
    emat = np.zeros(len(elems), int)
    names = sorted(setmap)
    for mi, nm in enumerate(names):
        for lab in setmap[nm]:
            if 1 <= lab <= len(emat):
                emat[lab - 1] = mi
    return nodes, np.asarray(elems), emat, names


def render_mesh(ax, path, title):
    nodes, elems, emat, names = load_mesh(path)
    polys = [nodes[e - 1] for e in elems]
    cmap = plt.get_cmap('tab10')
    pc = PolyCollection(polys, array=emat.astype(float), cmap=cmap,
                        edgecolors='k', linewidths=0.15)
    pc.set_clim(0, max(len(names) - 1, 1))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x2 (m, reference-axis frame)', fontsize=8)
    ax.set_ylabel('x3 (m)', fontsize=8)
    ax.tick_params(labelsize=7)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
render_mesh(axes[0], os.path.join(QUAD, 'iea_s00_solid.yaml'),
            'ROOT  eta=0.00  (near-circular, t/c=1.0)\npyNuMAD-quad: 2087 nodes, 1688 quads')
render_mesh(axes[1], os.path.join(QUAD, 'iea_s50_solid.yaml'),
            'TIP  eta=1.00  (sub-metre, TE sliver)\npyNuMAD-quad: 585 nodes, 474 quads')
fig.suptitle('The two cross-sections PreVABS cannot mesh, built cleanly by the pyNuMAD-style layered sweep',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig1_meshes.png'), dpi=150)
plt.close(fig)


# ------------------------------------------------------------- pipeline diagram
def box(ax, xy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02', fc=fc, ec='k', lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center', fontsize=8.2)


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle='-|>', mutation_scale=13, lw=1.3, color='0.3'))


fig, ax = plt.subplots(figsize=(11.5, 6.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.5); ax.axis('off')
ax.text(2.5, 10.1, 'PreVABS (current)', ha='center', fontsize=12, weight='bold')
ax.text(7.5, 10.1, 'pyNuMAD-style layered sweep', ha='center', fontsize=12, weight='bold')
BW, BH = 3.4, 0.9
P = [('Selig / UIUC airfoil .dat  (OML contour)', '#dbe9f6'),
     ('dividing points + per-segment layup', '#dbe9f6'),
     ('INWARD OFFSET each laminate\n(Clipper2 InflatePaths, EndType::Polygon)', '#f6c6c6'),
     ('DCEL faces (material, theta1/theta3)', '#dbe9f6'),
     ('gmsh 2-D mesh -> VABS .sg', '#dbe9f6')]
Q = [('OML contour + per-element laminate id', '#dbe9f6'),
     ('split OML into same-laminate RUNS', '#dbe9f6'),
     ('PER-VERTEX inward-normal SWEEP\n(1 quad layer per ply, half-thickness clamp)', '#c8e6c9'),
     ('weld coincident nodes + drop <4-corner slivers', '#dbe9f6'),
     ('2-D solid YAML (same schema as .sg->yaml)', '#dbe9f6')]
for col, seq in ((0.8, P), (5.8, Q)):
    y = 9.0
    for i, (t, fc) in enumerate(seq):
        box(ax, (col, y), BW, BH, t, fc)
        if i < len(seq) - 1:
            arrow(ax, col + BW / 2, y, y - 0.7)
        y -= 1.7
ax.text(0.8 + BW / 2, 9.0 - 2 * 1.7 - 0.35, 'FAILS: empty offset on thin wall / collapsed baseline',
        ha='center', va='top', fontsize=7.5, color='#b00000', style='italic')
ax.text(5.8 + BW / 2, 9.0 - 2 * 1.7 - 0.35, 'ROBUST: 1 offset vertex per base vertex -> never empty',
        ha='center', va='top', fontsize=7.5, color='#1b7a1b', style='italic')
fig.savefig(os.path.join(OUT, 'fig2_pipeline.png'), dpi=150, bbox_inches='tight')
plt.close(fig)


# ----------------------------------------------------- mechanism (why) diagram
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# a thin converging wedge (trailing-edge sliver): two outer walls meeting at TE
xw = np.linspace(0, 1, 40)
top = 0.14 * (1 - xw)          # upper wall
bot = -0.14 * (1 - xw)         # lower wall
tply = 0.10                    # laminate (offset) thickness

# LEFT: Clipper2 polygon inflate -> empty
ax = axes[0]
ax.plot(xw, top, 'k-', lw=2, label='OML wall')
ax.plot(xw, bot, 'k-', lw=2)
# inward offsets by tply (normal ~ +/- y): they CROSS before the TE -> region vanishes
ax.plot(xw, top - tply, color='#b00000', lw=1.6, ls='--')
ax.plot(xw, bot + tply, color='#b00000', lw=1.6, ls='--', label='inward offset (t)')
xc = xw[np.argmin(np.abs((top - tply) - (bot + tply)))]
ax.axvline(xc, color='0.6', lw=0.8, ls=':')
ax.fill_betweenx([-0.03, 0.03], xc, 1.0, color='#b00000', alpha=0.12)
ax.text(0.5, 0.19, 'PreVABS: Clipper2 InflatePaths', ha='center', fontsize=11, weight='bold')
ax.text(0.62, -0.20, 'offset > local half-thickness\n-> inset self-intersects\n-> InflatePaths returns EMPTY\n-> buildBaseOffsetMap FAILS',
        ha='center', va='top', fontsize=8.5, color='#b00000')
ax.set_aspect('equal'); ax.axis('off'); ax.legend(loc='upper left', fontsize=8)

# RIGHT: per-vertex normal sweep, clamped -> never empty
ax = axes[1]
ax.plot(xw, top, 'k-', lw=2)
ax.plot(xw, bot, 'k-', lw=2)
# per-vertex inward offset clamped to 0.49 * local half thickness
half = (top - bot) / 2
d = np.minimum(tply, 0.49 * half)
ax.plot(xw, top - d, color='#1b7a1b', lw=1.6)
ax.plot(xw, bot + d, color='#1b7a1b', lw=1.6)
for k in range(0, 40, 4):
    ax.annotate('', xy=(xw[k], top[k] - d[k]), xytext=(xw[k], top[k]),
                arrowprops=dict(arrowstyle='-|>', color='#1b7a1b', lw=0.9))
ax.text(0.5, 0.19, 'pyNuMAD / proposed C++: normal sweep', ha='center', fontsize=11, weight='bold')
ax.text(0.62, -0.20, 'each base vertex offset inward by\nmin(t, 0.49*half-thickness)\n-> 1 valid vertex per base vertex\n-> thin quads, NEVER empty',
        ha='center', va='top', fontsize=8.5, color='#1b7a1b')
ax.set_aspect('equal'); ax.axis('off')
fig.suptitle('Why the sweep is robust where the polygon inflate collapses (thin trailing-edge wall)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig3_mechanism.png'), dpi=150)
plt.close(fig)

print('wrote 3 figures to', OUT)
for f in ('fig1_meshes.png', 'fig2_pipeline.png', 'fig3_mechanism.png'):
    print(' ', f, os.path.getsize(os.path.join(OUT, f)), 'bytes')
