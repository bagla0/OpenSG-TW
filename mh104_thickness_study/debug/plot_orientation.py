"""Plot e1, e2, e3 element orientation directions read from a 1D-shell YAML (the mesh the JAX/FEniCS
shell actually consumes), to visually verify orientation is correct:
  e1 = beam axis  -> must be (0,0,1), out-of-plane (z)
  e2 = in-plane tangent -> follows the XML flow direction along the contour
  e3 = ply normal -> skin points OML->IML (toward centroid); BOTH webs consistent (matching solid).
Usage: python plot_orientation.py <yaml> [out.png]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

YAML = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(YAML)[0] + "_orient.png"


def _row(r):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)]


def _irow(r):
    return [int(float(v)) for v in (str(r[0]).split() if len(r) == 1 else r)]


d = yaml.safe_load(open(YAML))
nodes = np.array([_row(r) for r in d["nodes"]])               # (N,3)
elems = np.array([_irow(r) for r in d["elements"]]) - 1        # (E,2) 0-based
oris = np.array([_row(r) for r in d["elementOrientations"]])   # (E,9)
e1, e2, e3 = oris[:, 0:3], oris[:, 3:6], oris[:, 6:9]

P0, P1 = nodes[elems[:, 0], :2], nodes[elems[:, 1], :2]
ctr = 0.5 * (P0 + P1)

lp = np.full(len(elems), -1, int)
setnames = []
for si, s in enumerate(d["sets"]["element"]):
    setnames.append(s["name"])
    for lab in s["labels"]:
        lp[int(lab) - 1] = si
nset = len(setnames)
# web = the set whose elements are most vertical (|tangent_x| smallest), robust to set naming
vert = np.array([np.abs((P1 - P0)[lp == si, 0]).mean() / (np.linalg.norm((P1 - P0)[lp == si], axis=1).mean() + 1e-30)
                 if (lp == si).any() else 9 for si in range(nset)])
web_si = int(np.argmin(vert))

span = nodes[:, :2].max(0) - nodes[:, :2].min(0)
alen = 0.045   # fixed arrow length (~1/4 of the section thickness) so arrows don't overlap


def draw(ax, vec, title, mode="xy", zoom=None):
    for (a, b) in elems:
        ax.plot(nodes[[a, b], 0], nodes[[a, b], 1], color="0.82", lw=0.6, zorder=1)
    if mode == "xy":
        cmap = plt.get_cmap("tab10")
        for si in range(nset):
            m = lp == si
            col = "k" if si == web_si else cmap(si % 10)
            ax.quiver(ctr[m, 0], ctr[m, 1], vec[m, 0], vec[m, 1], color=col,
                      angles="xy", scale_units="xy", scale=1.0 / alen,
                      width=0.010 if si == web_si else 0.0032, zorder=3,
                      label=(setnames[si] + (" (web)" if si == web_si else "")))
        if zoom is None:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
    else:
        sc = ax.scatter(ctr[:, 0], ctr[:, 1], c=vec[:, 2], cmap="coolwarm", vmin=-1, vmax=1, s=14, zorder=3)
        plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.01, label="e1_z")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (chord)"); ax.set_ylabel("Y")
    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])


# zoom box around the rear (spar) web so the web e3 = -x and skin->web junction is legible
wx = ctr[lp == web_si, 0]
zx = float(np.median(wx[wx > np.median(wx)])) if (wx > np.median(wx)).any() else float(np.median(wx))
zoombox = (zx - 0.18, zx + 0.18, -0.13, 0.13)

fig, axs = plt.subplots(4, 1, figsize=(13, 12))
draw(axs[0], e2, "e2  (in-plane tangent / XML flow direction)")
draw(axs[1], e3, "e3  (ply normal: skin OML->IML toward centroid; both webs = -x, consistent)")
draw(axs[2], e3, "e3  ZOOM on rear web  (web arrows must point -x; skin arrows point inward)", zoom=zoombox)
draw(axs[3], e1, "e1  (beam axis = out-of-plane z; color = e1_z, expect +1 everywhere)", mode="z")
fig.suptitle(os.path.basename(YAML) + "  --  element orientation (e1,e2,e3) check", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
print("elements=%d  sets=%s  detected web set='%s'" % (len(elems), setnames, setnames[web_si]))
print("e1: all (0,0,1)?  e1_z in [%.3f, %.3f]" % (e1[:, 2].min(), e1[:, 2].max()))
print("e2.e3 orthogonal? max|e2.e3|=%.2e" % np.abs((e2 * e3).sum(1)).max())
print("e2x|e3 = +e1 (right-handed)? min cross_z=%.3f" % (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]).min())
print("web e3 x-sign(s):", np.unique(np.sign(np.round(e3[lp == web_si, 0], 6))))
