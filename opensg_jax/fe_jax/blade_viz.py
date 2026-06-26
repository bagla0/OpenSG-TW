"""Blade cross-section visualisation helpers (layup colouring + material orientation).

Globally-consistent layup colours: every cross-section is coloured by the *ply-stack composition*
(material + angle per ply, thickness ignored), so the same physical layup -- a carbon spar cap, a
foam-cored skin panel, a shear web -- gets the same colour in every station and every figure.

Public functions
----------------
build_layup_registry(shell_yamls)        -> (sig2idx, labels)   global layup registry
plot_layup_section(shell, solid, ...)    -> png   2-D solid + 1-D line mesh, coloured by layup, e2/e3 per region
plot_orientation_montage(shells, rRs...) -> png   grid of line cross-sections + one e2/e3 arrow per region
plot_span_loft(shells, rRs, ...)         -> png   line cross-sections lofted along the beam axis (3-D)

Convention: e2 = blue (in-plane ply direction), e3 = black (wall normal).
"""
import os
from collections import defaultdict, deque
import numpy as np
import yaml

from .orient_plot import _load
from .segment import read_solid_yaml

# layup colours -- deliberately avoid blue (reserved for the e2 arrow) and black (e3 arrow)
PALETTE = ["#2ca02c", "#ff7f0e", "#d62728", "#8c564b", "#9467bd", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]


def _sig(layup):
    return tuple((p[0], round(float(p[2]))) for p in layup)         # (material, angle) per ply; thickness ignored


def _label(layup):
    mats = [p[0] for p in layup]
    dom = max(layup, key=lambda p: float(p[1]))[0]                  # thickest ply = dominant material
    if "foam" in dom:
        return "foam (web)" if mats and mats[0] == "glass_biax" else "foam (skin)"
    return dom.replace("_", "-")


def build_layup_registry(shell_yamls):
    """Scan every shell YAML and assign each unique ply-stack a global colour index + material label."""
    reg, labels = {}, []
    for y in shell_yamls:
        raw = yaml.safe_load(open(y))
        sec = {s["elementSet"]: s["layup"] for s in raw["sections"]}
        for s in raw["sets"]["element"]:
            sg = _sig(sec[s["name"]])
            if sg not in reg:
                reg[sg] = len(labels); labels.append(_label(sec[s["name"]]))
    return reg, labels


def _station(shell_yaml, sig2idx):
    nd, elems, oris = _load(shell_yaml)
    raw = yaml.safe_load(open(shell_yaml))
    sec = {s["elementSet"]: s["layup"] for s in raw["sections"]}
    set2g = {s["name"]: sig2idx[_sig(sec[s["name"]])] for s in raw["sets"]["element"]}
    lay = np.full(len(elems), -1, int)
    for s in raw["sets"]["element"]:
        for lab in s["labels"]:
            lay[int(lab) - 1] = set2g[s["name"]]
    emid = np.array([nd[e[:2]].mean(0) for e in elems])
    return nd, elems, lay, oris, emid


def _regions(elems, sub):
    """connected components (sharing a node) among the element indices in ``sub``."""
    node2e = defaultdict(list)
    for k in sub:
        for n in elems[k][:2]:
            node2e[n].append(k)
    seen, comps = set(), []
    for k in sub:
        if k in seen:
            continue
        comp, dq = [], deque([k]); seen.add(k)
        while dq:
            e = dq.popleft(); comp.append(e)
            for n in elems[e][:2]:
                for e2 in node2e[n]:
                    if e2 not in seen:
                        seen.add(e2); dq.append(e2)
        comps.append(comp)
    return comps


def _region_arrows(ax, elems, lay, oris, emid, L):
    """one e2 (blue) / e3 (black) arrow at the representative element of every connected region."""
    for i in set(int(v) for v in lay[lay >= 0]):
        for comp in _regions(elems, np.where(lay == i)[0]):
            if len(comp) < 2:
                continue
            mids = emid[comp]; c = mids.mean(0)
            rep = comp[int(np.argmin(np.hypot(*(mids - c).T)))]
            x, y = emid[rep]; e2, e3 = oris[rep, 3:5], oris[rep, 6:8]
            ax.quiver(x, y, e3[0]*L, e3[1]*L, angles="xy", scale_units="xy", scale=1, color="black", width=0.004, zorder=7)
            ax.quiver(x, y, e2[0]*L, e2[1]*L, angles="xy", scale_units="xy", scale=1, color="tab:blue", width=0.004, zorder=6)


def _handles(used, labels):
    from matplotlib.patches import Patch
    return [Patch(color=PALETTE[i % len(PALETTE)], label=labels[i]) for i in sorted(used)]


def plot_layup_section(shell_yaml, solid_yaml, reg, out_png, arrow=0.22):
    """2-D solid mesh + 1-D shell LINE mesh, coloured by layup, with one e2/e3 arrow per region."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    from scipy.spatial import cKDTree
    sig2idx, labels = reg
    nd, elems, lay, oris, emid = _station(shell_yaml, sig2idx)
    sg = read_solid_yaml(solid_yaml)
    P, C = np.asarray(sg["points"])[:, :2], sg["cells"]
    cell_nn = lay[cKDTree(emid).query(np.array([P[c].mean(0) for c in C]))[1]]
    dom = np.asarray(sg["cell_domain_ids"])              # colour each solid DOMAIN by its majority layup (clean edges)
    solid_lay = np.empty(len(C), int)
    for d in np.unique(dom):
        m = dom == d
        solid_lay[m] = np.bincount(cell_nn[m]).argmax()
    fig, (axS, axL) = plt.subplots(2, 1, figsize=(12, 6.2))
    axS.add_collection(PolyCollection([P[c] for c in C], facecolors=[PALETTE[i % len(PALETTE)] for i in solid_lay], edgecolors="none"))
    axS.autoscale_view(); axS.set_aspect("equal"); axS.set_title("2-D solid mesh (%d elements)" % len(C))
    axS.set_ylabel("y3 (m)"); axS.set_xticklabels([])
    axL.add_collection(LineCollection([nd[e[:2]] for e in elems], colors=[PALETTE[i % len(PALETTE)] for i in lay], linewidths=2.0))
    _region_arrows(axL, elems, lay, oris, emid, arrow)
    axL.autoscale_view(); axL.set_aspect("equal")
    axL.set_title("1-D shell line mesh (%d elements)" % len(elems))
    axL.set_xlabel("y2 (m)"); axL.set_ylabel("y3 (m)")
    used = sorted(set(int(v) for v in lay[lay >= 0]))
    fig.legend(handles=_handles(used, labels), loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(used), fontsize=9, title="layup")
    fig.tight_layout(rect=[0, 0.06, 1, 1]); fig.savefig(out_png, dpi=145, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_orientation_montage(shell_yamls, rRs, reg, out_png, title="", arrow=0.06, ncol=2):
    """Grid of line cross-sections, each with one e2/e3 arrow per connected region."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    sig2idx, labels = reg
    n = len(shell_yamls); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.8 * nrow))
    axes = np.atleast_1d(axes).ravel(); used = set()
    for ax, y, r in zip(axes, shell_yamls, rRs):
        nd, elems, lay, oris, emid = _station(y, sig2idx); used |= set(int(v) for v in lay[lay >= 0])
        ax.add_collection(LineCollection([nd[e[:2]] for e in elems], colors=[PALETTE[i % len(PALETTE)] for i in lay], linewidths=1.4))
        _region_arrows(ax, elems, lay, oris, emid, arrow)
        ax.autoscale_view(); ax.set_aspect("equal"); ax.set_title("r = %.1f" % r, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.legend(handles=_handles(used, labels), loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=8.5, title="layup")
    fig.suptitle("line cross-section + material orientation   -   %s" % title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.86, 0.95]); fig.savefig(out_png, dpi=135, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_span_loft(shell_yamls, rs, reg, out_png):
    """Isometric loft: the line cross-sections placed at their span station ``r`` along the beam axis (x),
    coloured by layup, with a black beam axis through each section centre.  Orthographic (isometric)
    projection, no axes/grid/numbers."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    sig2idx, labels = reg
    fig = plt.figure(figsize=(13, 5.6)); ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    used, centers = set(), []
    for y, r in zip(shell_yamls, rs):
        nd, elems, lay, oris, emid = _station(y, sig2idx); used |= set(int(v) for v in lay[lay >= 0])
        segs = [[(r, nd[e[0], 0], nd[e[0], 1]), (r, nd[e[1], 0], nd[e[1], 1])] for e in elems]  # span = x
        ax.add_collection3d(Line3DCollection(segs, colors=[PALETTE[i % len(PALETTE)] for i in lay], linewidths=1.5))
        c = nd.mean(0); centers.append((r, c[0], c[1]))
    centers = np.array(centers)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="black", lw=1.8, zorder=10)   # beam axis through centres
    ax.set_xlim(min(rs) - 0.05, max(rs) + 0.05); ax.set_ylim(0, 5); ax.set_zlim(-0.9, 0.9)
    ax.set_box_aspect((6.5, 5.0, 1.6))
    ax.view_init(elev=18, azim=-74)
    ax.set_axis_off()
    fig.legend(handles=_handles(used, labels), loc="center right", fontsize=9, title="layup")
    fig.tight_layout(); fig.savefig(out_png, dpi=145, bbox_inches="tight"); plt.close(fig)
    return out_png
