"""Extract the OML boundary from the r=0.2 2-D solid yaml and define the dehom paths from the
OML coordinates: circumferential = upper (LP) OML LE->TE; cap-centre = OML->IML at the LP cap."""
import os, numpy as np, yaml
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D2 = os.path.expanduser("~/OpenSG-TW-claude/examples/data/2d_yaml")
SOLID = os.path.join(D2, "iea_r020_solid.yaml")
FIG = os.path.expanduser("~/claude_tmp")

ds = yaml.safe_load(open(SOLID))
def row(v): return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]
nodes = np.array([row(n)[:2] for n in ds["nodes"]])
tris = [[int(round(x)) - 1 for x in row(e)] for e in ds["elements"]]
mat = np.zeros(len(tris), int)
for grp in ds["sets"]["element"]:
    mi = int("".join(c for c in grp["name"] if c.isdigit()))
    for lab in grp["labels"]:
        mat[int(lab) - 1] = mi

# --- boundary edges (appear in exactly one triangle) ---
ec = defaultdict(int)
for t in tris:
    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
        ec[tuple(sorted((a, b)))] += 1
bnd = [e for e, c in ec.items() if c == 1]
adj = defaultdict(list)
for a, b in bnd:
    adj[a].append(b); adj[b].append(a)

# --- trace boundary loops ---
seen = set(); loops = []
for s in adj:
    if s in seen:
        continue
    loop = [s]; seen.add(s); cur = s; prev = None
    while True:
        nxts = [n for n in adj[cur] if n != prev and n not in seen]
        if not nxts:
            nxts = [n for n in adj[cur] if n != prev]           # close the loop
        if not nxts or nxts[0] == s:
            break
        prev, cur = cur, nxts[0]; loop.append(cur); seen.add(cur)
    if len(loop) > 5:
        loops.append(np.array(loop))
# OML = loop whose nodes span the widest x (contains LE + TE)
def span(lp): return np.ptp(nodes[lp, 0]) + np.ptp(nodes[lp, 1])
loops.sort(key=span, reverse=True)
oml = loops[0]; voids = loops[1:]
oml_xy = nodes[oml]
print("boundary loops: %d ; OML has %d nodes, span x[%.3f,%.3f] y[%.3f,%.3f]" %
      (len(loops), len(oml), oml_xy[:,0].min(), oml_xy[:,0].max(), oml_xy[:,1].min(), oml_xy[:,1].max()))

cen = nodes.mean(0)
# --- circumferential = UPPER (LP) OML nodes, LE->TE ---
up_full = oml_xy[oml_xy[:, 1] > cen[1]]
up_full = up_full[np.argsort(up_full[:, 0])]                   # LE (min x) -> TE (max x)
step = max(1, len(up_full) // 120)
up = up_full[::step]                                           # ~120 evenly-spaced OML nodes
print("circumferential: %d upper-OML nodes LE->TE (subsampled)" % len(up))

# --- cap-centre: STRAIGHT column NORMAL to the OML at the LP carbon cap ---
ecen = np.array([nodes[t].mean(0) for t in tris])
cap = ecen[(mat == 4) & (ecen[:, 1] > cen[1])]                 # Material_4 = carbon, upper
xc = float(np.median(cap[:, 0]))
j = int(np.argmin(np.abs(up_full[:, 0] - xc)))                 # OML cap node index
oml_pt = up_full[j]
a, b = max(0, j - 3), min(len(up_full) - 1, j + 3)
tang = up_full[b] - up_full[a]; tang = tang / np.linalg.norm(tang)   # local OML tangent
nrm = np.array([tang[1], -tang[0]])                            # perpendicular
if np.dot(cen - oml_pt, nrm) < 0:
    nrm = -nrm                                                 # point inward (toward centroid)
# wall thickness along the normal = first inner-void surface the ray crosses
vnodes = np.vstack([nodes[v] for v in voids]) if voids else np.empty((0, 2))
s = (vnodes - oml_pt) @ nrm                                    # along-normal distance
perp = np.linalg.norm((vnodes - oml_pt) - np.outer(s, nrm), axis=1)  # dist to the ray
cand = (s > 0.02) & (perp < 0.06)
L = float(s[cand].min()) if cand.any() else 0.096
cap_path = np.array([oml_pt + t * L * nrm for t in np.linspace(0, 1, 25)])   # STRAIGHT, normal
th = L
print("cap-centre: OML=(%.4f,%.4f)  inward-normal=(%.3f,%.3f)  wall thick=%.1f mm (straight, OML-normal)" %
      (oml_pt[0], oml_pt[1], nrm[0], nrm[1], th * 1e3))

def write_coords(path, pts):
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))]
    arc = 100.0 * z / z[-1] if z[-1] > 0 else np.zeros(len(pts))
    with open(path, "w") as f:
        for (x, y), a in zip(pts, arc):
            f.write("%.11f %.11f %.11f\n" % (x, y, a))
    print("wrote", os.path.basename(path), "(%d pts)" % len(pts))

write_coords(os.path.join(D2, "solid.lp_sparcap_center_thickness_r020.coords"), cap_path)
write_coords(os.path.join(D2, "solid.circumferential_r020.coords"), up)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(oml_xy[:, 0], oml_xy[:, 1], ".", color="0.7", ms=2, label="OML boundary")
for v in voids:
    ax.plot(nodes[v, 0], nodes[v, 1], ".", color="0.85", ms=1)
ax.plot(up[:, 0], up[:, 1], "-", color="#1f77b4", lw=2, label="circumferential (upper OML)")
ax.plot(cap_path[:, 0], cap_path[:, 1], "-o", color="#d62728", ms=3, lw=2, label="LP cap OML->IML")
ax.set_aspect("equal"); ax.legend(loc="lower right"); ax.set_title("IEA r=0.2 OML-based dehom paths")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "r020_oml_paths.png"), dpi=140, bbox_inches="tight")
print("wrote r020_oml_paths.png")
