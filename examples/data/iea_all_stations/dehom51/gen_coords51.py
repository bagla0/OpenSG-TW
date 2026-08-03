'''gen_coords51.py -- build the r0.2 (iea_s10) dehom .coords from the VABS .SM GAUSS POINTS so all
three solvers (VABS / JAX-solid / RM) sample identical points.  Two cross-section paths for now:
  circumferential : OML LP surface LE->TE (near-surface gauss points)
  lp_sparcap_left : through-thickness column NORMAL to OML at the LEFT edge of the LP carbon cap
                    (the S12-critical cap/web-junction path)

FRAME: the .sg / .SM (VABS) are LE-based (x from 0); the RM shell + solid yaml are reference-axis
(x shifted by ~ -section_offset). We write the .coords in the VABS (LE) frame and record the shift
DX (ref-axis = LE - DX) so the RM driver can project correctly.  columns: y2 y3 arc%  (VABS frame).
'''
import os
import numpy as np
from collections import defaultdict

ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
SG = os.path.join(ROOT, 'shell51/sg_v201/iea_s10.sg')
SM = os.path.join(ROOT, 'dehom_iea/sg_v201/iea_s10.sg.SM')
SHELL_YAML = os.path.join(ROOT, 'shell51/1d_yaml/iea_s10_shell.yaml')
OUT = os.path.join(ROOT, 'dehom51', 'coords')
os.makedirs(OUT, exist_ok=True)


# ---------------- parse the .sg (LE frame): nodes, tris, per-elem material, material E1 ----------
def parse_sg(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    h = next(i for i, l in enumerate(L)
             if len(l.split()) == 3 and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
    nn, ne, nm = [int(x) for x in L[h].split()]
    xy = np.array([[float(L[h + 1 + k].split()[1]), float(L[h + 1 + k].split()[2])] for k in range(nn)])
    conn = [[int(x) for x in L[h + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
    p = h + 1 + nn + ne                                            # block A: eid group theta1
    egrp = np.array([int(L[p + k].split()[1]) for k in range(ne)])
    p2 = p + ne                                                   # group block: gid matid theta3
    gmat = {int(L[p2 + k].split()[0]): int(L[p2 + k].split()[1]) for k in range(nm)}
    p3 = p2 + nm                                                  # material blocks (5 lines each)
    E1 = {}
    for m in range(nm):
        mid = int(L[p3 + 5 * m].split()[0])
        E1[mid] = float(L[p3 + 5 * m + 1].split()[0])             # first elastic constant = E1
    emat = np.array([gmat[g] for g in egrp])                      # per-element material id
    return xy, conn, emat, E1


xy, conn, emat, E1 = parse_sg(SG)
ecen = np.array([xy[[n - 1 for n in c]].mean(0) for c in conn])
carbon = max(E1, key=E1.get)                                     # carbon = highest E1
cen = xy.mean(0)
print('sg: %d nodes %d elems ; carbon = material %d (E1=%.2e)' % (len(xy), len(conn), carbon, E1[carbon]))
print('LE-frame bbox x[%.3f,%.3f] y[%.3f,%.3f]  centroid=(%.3f,%.3f)'
      % (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max(), cen[0], cen[1]))

# frame shift to ref-axis: LE min-x maps to ref-axis min-x; DX = LE_LEx - refaxis_LEx
import yaml
sh = yaml.safe_load(open(SHELL_YAML))
srow = lambda v: [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]
snd = np.array([srow(n)[:2] for n in sh['nodes']])
DX = xy[:, 0].min() - snd[:, 0].min()                            # x_LE = x_refaxis + DX
print('ref-axis shift DX = %.5f m  (x_refaxis = x_LE - DX)  [shell x range %.3f..%.3f]'
      % (DX, snd[:, 0].min(), snd[:, 0].max()))

# ---------------- .SM gauss points (LE frame) ----------------
d = np.loadtxt(SM, skiprows=2)
gp = d[:, :2]                                                    # gauss-point coords (LE frame)
print('.SM: %d gauss points  x[%.3f,%.3f] y[%.3f,%.3f]' % (len(gp), gp[:, 0].min(), gp[:, 0].max(),
                                                            gp[:, 1].min(), gp[:, 1].max()))


def write_coords(name, pts):
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))]
    arc = 100.0 * z / z[-1] if z[-1] > 0 else np.zeros(len(pts))
    p = os.path.join(OUT, name)
    with open(p, 'w') as f:
        for (x, y), a in zip(pts, arc):
            f.write('%.11f %.11f %.11f\n' % (x, y, a))
    print('wrote %s (%d pts, arc %.1f mm)' % (name, len(pts), z[-1] * 1e3))


# ---------------- OML boundary (LP surface) from the .sg tri mesh ----------------
ec = defaultdict(int)
for c in conn:
    if len(c) == 3:
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[0])):
            ec[tuple(sorted((a, b)))] += 1
bnd = [e for e, k in ec.items() if k == 1]
adj = defaultdict(list)
for a, b in bnd:
    adj[a].append(b); adj[b].append(a)
seen = set(); loops = []
for s0 in adj:
    if s0 in seen:
        continue
    loop = [s0]; seen.add(s0); cur = s0; prev = None
    while True:
        nx = [n for n in adj[cur] if n != prev and n not in seen] or [n for n in adj[cur] if n != prev]
        if not nx or nx[0] == s0:
            break
        prev, cur = cur, nx[0]; loop.append(cur); seen.add(cur)
    if len(loop) > 5:
        loops.append(np.array(loop))
loops.sort(key=lambda lp: np.ptp(xy[lp, 0]) + np.ptp(xy[lp, 1]), reverse=True)
oml = loops[0]; voids = loops[1:]
oml_xy = xy[oml - 1] if oml.min() >= 1 else xy[oml]              # node ids are 1-based in conn
# conn nodes are 1-based -> oml holds 1-based ids; index xy with id-1
oml_xy = xy[[n - 1 for n in oml]]
up = oml_xy[oml_xy[:, 1] > cen[1]]
up = up[np.argsort(up[:, 0])]                                   # LE->TE
print('OML loops=%d ; LP-surface nodes=%d' % (len(loops), len(up)))

# ---------------- (1) circumferential: near-surface gauss points along LP OML, LE->TE ----------------
# snap each subsampled OML node to its nearest .SM gauss point (near-surface, VABS-exact)
from scipy.spatial import cKDTree
tree = cKDTree(gp)
step = max(1, len(up) // 120)
circ_nodes = up[::step]
_, idx = tree.query(circ_nodes)
circ = gp[idx]
circ = circ[np.argsort(circ[:, 0])]
write_coords('iea_s10.circumferential.coords', circ)

# ---------------- (2) LEFT LP spar-cap through-thickness (S12-critical) ----------------
capmask = (emat == carbon) & (ecen[:, 1] > cen[1])              # LP carbon cap elements
capx = ecen[capmask, 0]
xL = float(np.percentile(capx, 8))                              # LEFT edge (near the fwd web junction)
band = 8e-4
col_m = (np.abs(gp[:, 0] - xL) < band) & (gp[:, 1] > ecen[capmask, 1].min() - 0.02)
col = gp[col_m]
col = col[np.argsort(-col[:, 1])]                              # OML (high y) -> IML
keep, last = [], None
for i in range(len(col)):
    if last is None or abs(col[i, 1] - last) > 2e-4:
        keep.append(i); last = col[i, 1]
    elif abs(col[i, 0] - xL) < abs(col[keep[-1], 0] - xL):
        keep[-1] = i
col = col[keep]
print('LEFT cap column xL=%.4f (cap x range %.3f..%.3f)  %d gauss pts, depth %.1f mm'
      % (xL, capx.min(), capx.max(), len(col), 1e3 * (col[0, 1] - col[-1, 1]) if len(col) else 0))
write_coords('iea_s10.lp_sparcap_left_thickness.coords', col)

# record the frame shift for the RM driver
np.savetxt(os.path.join(OUT, 'frame_shift.dat'), [DX],
           header='DX (m): x_refaxis = x_LE - DX ; VABS/.SM/.coords are LE frame, RM shell is ref-axis')
print('\nwrote coords/ + frame_shift.dat (DX=%.5f)' % DX)
