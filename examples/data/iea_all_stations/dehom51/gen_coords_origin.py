'''gen_coords_origin.py -- regenerate the r0.2 (iea_s10) dehom .coords DIRECTLY from the (0,0)-origin
2-D solid yaml, so the paths live in the reference-axis frame (same frame as the RM 1-D shell yaml).
No LE->ref shift is needed any more (frame_shift DX = 0).

Two cross-section paths (same definitions as before):
  circumferential            : OML LP (suction) surface nodes, LE -> TE
  lp_sparcap_left_thickness  : through-thickness column of element centroids at the LEFT edge of the
                               LP carbon spar cap (the S12-critical cap/web-junction path)

Also writes a verification PNG (2-D solid colored by material + both paths) so the coords can be
checked against the actual section.  coords columns: y2 y3 arc%   (all in the (0,0) frame).
'''
import os
import numpy as np
import yaml
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
SOL = os.path.join(ROOT, 'shell51/2d_yaml/iea_s10_solid.yaml')          # (0,0)-origin 2-D solid mesh
OUT = os.path.join(ROOT, 'dehom51', 'coords')
os.makedirs(OUT, exist_ok=True)


def frow(v):
    if isinstance(v, str):
        return [float(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [float(x) for x in v[0].split()]
    return [float(x) for x in v]


def irow(v):
    if isinstance(v, str):
        return [int(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [int(x) for x in v[0].split()]
    return [int(x) for x in v]


# ---------------- parse the (0,0) 2-D solid yaml ----------------
y = yaml.safe_load(open(SOL))
xy = np.array([frow(n)[:2] for n in y['nodes']])                       # node coords (0,0 frame)
conn = [irow(e) for e in y['elements']]                                # node-id lists
nid_min = min(min(c) for c in conn)                                    # 0- or 1-based?
base = nid_min                                                         # subtract to get 0-based index
conn = [[n - base for n in c] for c in conn]
ne = len(conn)

# per-element material id from sets['element'] (each set = one material region)
emat = np.zeros(ne, dtype=int)
matnames = [m['name'] for m in y['materials']]
E1 = np.array([frow(m['E'])[0] for m in y['materials']])               # first elastic const per material
setlist = y['sets']['element']                                         # list of {name, labels}
lab_min = min(min(s['labels']) for s in setlist)                       # element-label base (0/1)
for s in setlist:
    mi = matnames.index(s['name']) if s['name'] in matnames else 0     # map set -> material index
    for lab in s['labels']:
        emat[lab - lab_min] = mi
carbon = int(np.argmax(E1))                                            # carbon = highest E1 material index
import json                                                            # real material names (Material_N -> name)
_mnf = os.path.join(OUT, 'material_names.json')
_mnames = json.load(open(_mnf)) if os.path.exists(_mnf) else {}
realname = lambda mi: _mnames.get(matnames[mi], matnames[mi])
ecen = np.array([xy[c].mean(0) for c in conn])
cen = xy.mean(0)
print('solid yaml: %d nodes  %d tris  %d materials' % (len(xy), ne, len(matnames)))
print('  node-id base=%d  elem-label base=%d' % (base, lab_min))
print('  bbox x[%.3f, %.3f] y[%.3f, %.3f]  centroid=(%.3f, %.3f)'
      % (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max(), cen[0], cen[1]))
print('  carbon = %s (idx %d, E1=%.2e)' % (matnames[carbon], carbon, E1[carbon]))


def write_coords(name, pts):
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))]
    arc = 100.0 * z / z[-1] if z[-1] > 0 else np.zeros(len(pts))
    with open(os.path.join(OUT, name), 'w') as f:
        for (x, yv), a in zip(pts, arc):
            f.write('%.11f %.11f %.11f\n' % (x, yv, a))
    print('  wrote %s (%d pts, arc %.1f mm)' % (name, len(pts), z[-1] * 1e3))
    return z[-1]


# ---------------- OML boundary loop from the tri mesh ----------------
ec = defaultdict(int)
for c in conn:
    if len(c) == 3:
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[0])):
            ec[tuple(sorted((a, b)))] += 1
bnd = [e for e, k in ec.items() if k == 1]                             # boundary edges (appear once)
adj = defaultdict(list)
for a, b in bnd:
    adj[a].append(b); adj[b].append(a)
seen = set(); loops = []
for s0 in adj:
    if s0 in seen:
        continue
    loop = [s0]; seen.add(s0); cur = s0; prev = None
    while True:
        nxt = [n for n in adj[cur] if n != prev and n not in seen] or [n for n in adj[cur] if n != prev]
        if not nxt or nxt[0] == s0:
            break
        prev, cur = cur, nxt[0]; loop.append(cur); seen.add(cur)
    if len(loop) > 5:
        loops.append(np.array(loop))
loops.sort(key=lambda lp: np.ptp(xy[lp, 0]) + np.ptp(xy[lp, 1]), reverse=True)
oml = loops[0]
oml_xy = xy[oml]
up = oml_xy[oml_xy[:, 1] > cen[1]]                                     # LP (suction/upper) surface
up = up[np.argsort(up[:, 0])]                                         # LE (min x) -> TE (max x)
print('OML: %d boundary loops ; LP-surface nodes=%d' % (len(loops), len(up)))

# ---------------- VABS .SM gauss points : the ACTUAL path sample points (0,0 frame) ----------------
# the path is built on the VABS gauss-point coordinates so RM + VABS evaluate identical points.
from scipy.spatial import cKDTree
SM = os.path.join(ROOT, 'dehom51', 'out', 'VABS_iea51', 'iea_s10.sg.SM')
gp = np.loadtxt(SM, skiprows=2)[:, :2]                                # VABS gauss-point coords, (0,0)
gtree = cKDTree(gp)
print('VABS .SM gauss points: %d  x[%.3f, %.3f] y[%.3f, %.3f]'
      % (len(gp), gp[:, 0].min(), gp[:, 0].max(), gp[:, 1].min(), gp[:, 1].max()))

# ---------------- (1) circumferential : LP OML gauss points, LE->TE ----------------
# subsample the OML LP nodes, snap each to its nearest VABS gauss point (near-surface, VABS-exact)
step = max(1, len(up) // 120)
_, idx = gtree.query(up[::step])
circ = gp[idx]
circ = circ[np.argsort(circ[:, 0])]                                  # LE (min x) -> TE (max x)
arc_c = write_coords('iea_s10.circumferential.coords', circ)

# ---------------- (2) LEFT LP spar-cap through-thickness ----------------
capmask = (emat == carbon) & (ecen[:, 1] > cen[1])                    # LP carbon cap elements
capx = ecen[capmask, 0]; capymin = ecen[capmask, 1].min()
xL = float(np.percentile(capx, 50))                                  # cap CENTER: clean single-wall column
band = 8.0e-4                                                        # (the left edge is a cap/sandwich T-junction)
col_m = (np.abs(gp[:, 0] - xL) < band) & (gp[:, 1] > capymin - 0.02) & (gp[:, 1] > cen[1])
col = gp[col_m]                                                       # VABS gauss points in the column
col = col[np.argsort(-col[:, 1])]                                     # OML (high y) -> IML (low y)
keep, last = [], None
for i in range(len(col)):
    if last is None or abs(col[i, 1] - last) > 2e-4:
        keep.append(i); last = col[i, 1]
    elif abs(col[i, 0] - xL) < abs(col[keep[-1], 0] - xL):
        keep[-1] = i
col = col[keep]
print('LEFT cap column xL=%.4f (cap x[%.3f, %.3f])  %d pts, depth %.1f mm'
      % (xL, capx.min(), capx.max(), len(col), 1e3 * (col[0, 1] - col[-1, 1]) if len(col) else 0))
write_coords('iea_s10.lp_sparcap_left_thickness.coords', col)

# frame shift is now zero (coords already in the (0,0) ref-axis frame)
np.savetxt(os.path.join(OUT, 'frame_shift.dat'), [0.0],
           header='DX (m)=0 : coords are already in the (0,0) reference-axis frame (from 2D solid yaml)')

# ---------------- verification PNG : material-colored section + the 2 paths ----------------
polys = [xy[c] for c in conn]
cmap = plt.cm.tab10
fig, ax = plt.subplots(figsize=(13, 5))
ax.add_collection(PolyCollection(polys, facecolors=[cmap(emat[k] % 10) for k in range(ne)],
                                 edgecolors='none'))
# material legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=cmap(mi % 10), label='%s%s' % (realname(mi), '  (carbon)' if mi == carbon else ''))
           for mi in range(len(matnames))]
ax.plot(circ[:, 0], circ[:, 1], '-', color='k', lw=2.8, label='circumferential (LP OML, LE→TE)')
# spar-cap path in CYAN (not magenta): the cap material is red, so a red/pink path was invisible on it
ax.plot(col[:, 0], col[:, 1], '-o', color='cyan', mec='k', mew=0.7, ms=8, lw=4.0, label='lp_sparcap_left (thickness)')
ax.plot([0], [0], 'w+', mec='k', mew=1.5, ms=14)                      # (0,0) reference axis
ax.set_aspect('equal'); ax.autoscale(); ax.axis('off')
lg1 = ax.legend(loc='lower right', fontsize=13, framealpha=0.95)      # paths -> bottom-right, clear of curves
ax.add_artist(lg1)
ax.legend(handles=handles, loc='upper right', fontsize=12, title='materials', framealpha=0.95)
fig.tight_layout()
png = os.path.join(ROOT, 'dehom51', 'out', 'r020_section_paths_origin.png')
os.makedirs(os.path.dirname(png), exist_ok=True)
fig.savefig(png, dpi=160, bbox_inches='tight'); plt.close(fig)
print('wrote verification PNG -> out/r020_section_paths_origin.png')
print('DONE: coords regenerated in the (0,0) frame (frame_shift DX=0).')
