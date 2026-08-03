"""sg_census.py -- material census of the VABS .sg at the web bands vs the shell-yaml layups.

For each web (x2 band around the RM web line) and for the spar-cap band, sum tri areas per
material and divide by the wall run length -> implied per-material thickness in the .sg.
Compare with the OpenSG shell yaml plies. Decides whether the two models carry the SAME
wall laminate (registration offset only) or DIFFERENT laminates.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
VABS = os.path.join(IEA, "out", "VABS_iea51")
SG = os.path.join(VABS, "iea_s10.sg")

L = [l for l in open(SG).read().splitlines() if l.strip()]
hi = next(i for i, l in enumerate(L) if len(l.split()) == 3
          and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
nn, ne, nm = [int(x) for x in L[hi].split()]
print(".sg: %d nodes, %d elems, %d materials" % (nn, ne, nm))
nodes = np.array([[float(x) for x in L[hi + 1 + k].split()[1:3]] for k in range(nn)])
conn = [[int(x) for x in L[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
lay = np.zeros(ne, int)
th1 = np.zeros(ne)
for k in range(ne):
    p = L[hi + 1 + nn + ne + k].split()
    lay[k] = int(float(p[1])); th1[k] = float(p[2])

# layer table: layer_id material_id theta3  (nm lines follow the element theta block)
tail = hi + 1 + nn + 2 * ne
lyr2mat = {}
k = tail
while k < len(L):
    p = L[k].split()
    if len(p) == 3 and p[0].lstrip('-').isdigit():
        lyr2mat[int(p[0])] = int(p[1]); k += 1
    else:
        break
print("layer->material:", lyr2mat)
# material blocks: "id iso" then property lines; grab first stiffness number per material
matE = {}
while k < len(L):
    p = L[k].split()
    if len(p) == 2 and p[0].isdigit():
        mid = int(p[0])
        e1 = float(L[k + 1].split()[0])
        matE[mid] = e1
        k += 1
        while k < len(L) and len(L[k].split()) != 2:
            k += 1
    else:
        k += 1
print("material E1:", {m: '%.3g' % e for m, e in sorted(matE.items())})

cen = np.zeros((ne, 2)); area = np.zeros(ne)
for k, c in enumerate(conn):
    c0 = [n - 1 for n in c]
    P = nodes[c0]
    cen[k] = P.mean(0)
    if len(c0) == 3:
        area[k] = 0.5 * abs(np.cross(P[1] - P[0], P[2] - P[0]))
    else:
        area[k] = 0.5 * abs(np.cross(P[1] - P[0], P[2] - P[0])) + 0.5 * abs(np.cross(P[2] - P[0], P[3] - P[0]))

mat_e = np.array([lyr2mat.get(int(l), -1) for l in lay])

# webs: vertical bands (from web_probe: x2 ~ -1.??, -0.044, +1.964 -- recompute from geometry:
# interior elements = far from outer boundary loop)
from collections import defaultdict
ec = defaultdict(int)
tris_all = []
for c in conn:
    c0 = [n - 1 for n in c]
    t = c0[:3]
    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
        ec[(min(a, b), max(a, b))] += 1
# boundary loop of the TRIANGLE soup is approximate for quads; use node-hull distance instead:
from scipy.spatial import cKDTree
# outer loop: nodes on the convex-ish outer boundary -- reuse edge-count on tri soup
adj = defaultdict(list)
for (a, b), c in ec.items():
    if c == 1:
        adj[a].append(b); adj[b].append(a)
loops, seen = [], set()
for s in adj:
    if s in seen:
        continue
    loop, cur, prev = [s], s, -1
    seen.add(s)
    while True:
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        if cur == s:
            break
        loop.append(cur); seen.add(cur)
    loops.append(loop)
loops.sort(key=lambda l: -len(l))
outer = np.array(loops[0])
kd = cKDTree(nodes[outer])
d_out = kd.query(cen)[0]
webm = d_out > 0.12

xw_list = []
wcen = cen[webm]
xs = np.sort(wcen[:, 0])
# cluster web x2 positions
splits = np.where(np.diff(xs) > 0.4)[0]
groups = np.split(xs, splits + 1)
for g in groups:
    if len(g) > 50:
        xw_list.append(float(np.median(g)))
print("web x2 clusters:", ["%.3f" % x for x in xw_list])

def census(mask, name, run_len):
    tot = area[mask].sum()
    print("\n== %s: %d elems, total area %.5f m^2, run length %.3f m -> wall t %.1f mm" %
          (name, mask.sum(), tot, run_len, 1e3 * tot / run_len))
    for m in sorted(set(mat_e[mask])):
        a = area[mask & (mat_e == m)].sum()
        print("   mat %d (E1 %.3g): area %.5f -> t %.2f mm  (%.1f%%)"
              % (m, matE.get(m, 0), a, 1e3 * a / run_len, 100 * a / tot))

for i, xw in enumerate(xw_list):
    mm = webm & (np.abs(cen[:, 0] - xw) < 0.30)
    ys = cen[mm][:, 1]
    census(mm, "WEB %d (x2~%.3f)" % (i, xw), float(ys.max() - ys.min()))

# spar cap band (suction side, near max thickness): x2 in [-0.5, 0.5], y3 > 0, skin only
capm = (~webm) & (np.abs(cen[:, 0]) < 0.5) & (cen[:, 1] > 0)
census(capm, "CAP band SS x2 in [-0.5,0.5]", 1.0)
print("\nyaml: cap layup_2 = gelcoat 0.5 / triax 3.0 / carbon 81.1 / triax 3.0 mm (t=87.6)")
print("yaml: web layup_5 = biax 2.0 / foam 42.0 / biax 2.0 mm (t=46.0)")
