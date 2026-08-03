"""spanwise_sections51.py -- one PNG of ALL 51 IEA-22 shell (1-D) cross-sections stacked along the span.

Each of the 51 stations' shell YAML (shell51/1d_yaml/iea_sNN_shell.yaml) is drawn in 3-D at its physical
span location x1 = (i/50)*BLADE_LEN: the OML skin loop (blue) + the shear webs (red).  The section
origins (the (0,0) reference axis of every station) are joined by a dotted line with a red dot at each
station node -- the straight beam reference axis the 1-D SGs share.
Output: out/spanwise_sections/spanwise_sections51.png
"""
import os
from collections import defaultdict
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations"
YDIR = os.path.join(HERE, "shell51", "1d_yaml")
OUT = os.path.join(HERE, "dehom51", "out", "spanwise_sections"); os.makedirs(OUT, exist_ok=True)
BLADE_LEN = 138.204


def _row(v):
    if isinstance(v, str):
        return [float(x) for x in v.split()]
    if isinstance(v, list) and len(v) and isinstance(v[0], str):
        return [float(x) for x in v[0].split()]
    return [float(x) for x in v]


def outer_loop(nd, el):
    """Walk the outer skin loop (leftmost start, consistent left turn) so webs = the rest."""
    adj = defaultdict(set)
    for e in el:
        a, b = int(e[0]), int(e[1]); adj[a].add(b); adj[b].add(a)
    xy = nd[:, :2]
    start = int(np.argmin(xy[:, 0])) + 1
    prev, cur = None, start; din = np.array([0.0, -1.0]); loop = []
    for _ in range(2 * len(el) + 5):
        v = xy[cur - 1]
        cand = [w for w in adj[cur] if w != prev and w != cur]
        if not cand:
            break
        angs = [np.arctan2(din[0] * (xy[w - 1] - v)[1] - din[1] * (xy[w - 1] - v)[0],
                           din[0] * (xy[w - 1] - v)[0] + din[1] * (xy[w - 1] - v)[1]) for w in cand]
        nxt = cand[int(np.argmin(angs))]
        loop.append(cur); prev = cur; din = xy[nxt - 1] - v; cur = nxt
        if cur == start:
            break
    return np.array(loop)


fig = plt.figure(figsize=(17, 7))
ax = fig.add_subplot(111, projection="3d")
origins = []; nweb = 0; nok = 0
for i in range(51):
    f = os.path.join(YDIR, "iea_s%02d_shell.yaml" % i)
    if not os.path.exists(f):
        continue
    d = yaml.safe_load(open(f))
    nd = np.array([_row(n)[:2] for n in d["nodes"]])
    el = [[int(x) for x in _row(e)] for e in d["elements"]]
    X = (i / 50.0) * BLADE_LEN
    lp = outer_loop(nd, el)
    if len(lp) > 3:
        c = nd[lp - 1]; cc = np.vstack([c, c[0]])
        ax.plot(np.full(len(cc), X), cc[:, 0], cc[:, 1], "-", color="#1f77b4", lw=0.7)
    onloop = set(int(x) for x in lp)
    for e in el:
        a, b = int(e[0]), int(e[1])
        if not (a in onloop and b in onloop):
            ax.plot([X, X], [nd[a - 1, 0], nd[b - 1, 0]], [nd[a - 1, 1], nd[b - 1, 1]],
                    "-", color="#d62728", lw=0.8)
            nweb += 1
    origins.append([X, 0.0, 0.0]); nok += 1

origins = np.array(origins)
ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], ":", color="k", lw=1.8,
        label="reference axis (section origins)")
ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], color="red", s=22, depthshade=False,
           label="station nodes")
ax.set_xlabel("span  $x_1$  [m]", labelpad=14, fontsize=12)
ax.set_ylabel("chord  $y_2$  [m]", fontsize=12); ax.set_zlabel("flap  $y_3$  [m]", fontsize=12)
ax.view_init(elev=16, azim=-72)
ax.set_box_aspect((6.5, 1.7, 0.9))
ax.legend(loc="upper left", fontsize=11)
fig.tight_layout()
png = os.path.join(OUT, "spanwise_sections51.png")
fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)
print("wrote %s  (%d/51 sections, %d web segments drawn)" % (png, nok, nweb))
