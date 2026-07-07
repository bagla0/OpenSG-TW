"""verify_boundary_mesh.py -- prove the ACTUAL solid mesh used for the ell3w results
is a single connected (watertight) component, and render its TRUE end face (extracted
from the mesh, not sketched) alongside the shell ring.
"""
import os, sys, math
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
SOL = os.path.join(HERE, "out", "ell3w_thick", "solid", "solid_e3w.yaml")   # THE thick-boundary results mesh
SHL = os.path.join(HERE, "out", "ell3w_thick", "shell", "shell_e3w.yaml")


def load_solid(fn):
    d = yaml.load(open(fn), Loader=CL)
    nodes = np.array([[float(v) for v in (r[0] if isinstance(r, list) else r).split()]
                      for r in d["nodes"]])
    hexes = np.array([[int(v) for v in (r[0] if isinstance(r, list) else r).split()]
                      for r in d["elements"]]) - 1
    return nodes, hexes


nodes, hexes = load_solid(SOL)
N, M = len(nodes), len(hexes)
used = np.unique(hexes)

# 1) connectivity: union-find over hex node adjacency
parent = np.arange(N)
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for h in hexes:
    r0 = find(h[0])
    for k in h[1:]:
        parent[find(k)] = r0
roots = set(int(find(u)) for u in used)
# 2) duplicate (coincident but distinct) nodes among used nodes -> should be 0
from scipy.spatial import cKDTree
uc = nodes[used]
pairs = cKDTree(uc).query_pairs(1e-8)
print("SOLID mesh actually used for ell3w results: %s" % SOL)
print("  nodes=%d  hexes=%d  used-nodes=%d" % (N, M, len(used)))
print("  connected components over used nodes: %d   (1 = watertight)" % len(roots))
print("  coincident-but-distinct node pairs (<1e-8): %d   (0 = webs share skin nodes)" % len(pairs))

# 3) render the TRUE end face: quad faces of hexes lying on z=min plane
z = nodes[:, 2]; z0 = z.min(); tol = 1e-6 * (z.max() - z0 + 1)
faces = []
for h in hexes:
    onz = [n for n in h if z[n] - z0 < tol]
    if len(onz) == 4:
        p = nodes[onz][:, :2]
        c = p.mean(0); ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
        faces.append(p[np.argsort(ang)])
print("  end-face quads at z=%.3f: %d" % (z0, len(faces)))

d = yaml.load(open(SHL), Loader=CL)
snodes = np.array(d["nodes"], float); squads = np.array(d["elements"], int) - 1
zs = snodes[:, 2]; zs0 = zs.min()
ring_edges = []
for q in squads:
    on = [n for n in q if zs[n] - zs0 < tol]
    if len(on) == 2:
        ring_edges.append(snodes[on][:, :2])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2))
for e in ring_edges:
    axL.plot(e[:, 0], e[:, 1], "-", color="#4c78a8", lw=1.3)
rn = np.unique(np.concatenate([e for e in ring_edges]), axis=0)
axL.plot(rn[:, 0], rn[:, 1], "o", color="#26456e", ms=3)
axL.set_title("RM shell ring (mid-surface)", fontsize=11)
axR.add_collection(PolyCollection(faces, facecolors="#e9b98a",
                                  edgecolors="#7a4a1e", linewidths=0.5))
axR.set_title("3-D solid end face (extracted from the mesh)", fontsize=11)
for ax in (axL, axR):
    ax.set_aspect("equal"); ax.autoscale(); ax.set_xlabel("$x_2$"); ax.set_ylabel("$x_3$")
fig.tight_layout()
fn = os.path.join(HERE, "ell_boundary_true.png")
fig.savefig(fn, dpi=170, bbox_inches="tight"); plt.close(fig)
print("wrote", fn)
