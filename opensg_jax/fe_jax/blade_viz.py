"""Blade cross-section visualisation helpers (material distribution + layup + orientation).

- The **2-D solid** mesh is coloured by **material** (the PreVABS mesh resolves every ply, so the through-
  thickness layers show): each ``cell_domain_ids`` domain is matched to a named material by its E1.
- The **1-D shell** line mesh is coloured by **layup** (the wall laminate) keyed by ply-stack composition,
  so the same laminate -- carbon spar cap, foam skin panel, foam shear web, glass-UD edge -- is one colour
  in every station and figure.

Convention: e2 = blue (in-plane ply direction), e3 = black (wall normal).
"""
import os
from collections import defaultdict, deque
import numpy as np
import yaml

from .orient_plot import _load
from .segment import read_solid_yaml

# layup colours -- avoid blue (e2 arrow) and black (e3 arrow)
PALETTE = ["#2ca02c", "#ff7f0e", "#d62728", "#8c564b", "#9467bd", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
# fixed material colours (foam is the light-grey bulk; carbon stands out red)
MAT_PALETTE = {"medium_density_foam": "#d9d9d9", "gelcoat": "#fdbf6f", "glass_triax": "#a6cee3",
               "glass_uniax": "#1f78b4", "carbon_uniax": "#e31a1c", "glass_biax": "#33a02c"}
MAT_ORDER = ["gelcoat", "glass_triax", "glass_biax", "glass_uniax", "medium_density_foam", "carbon_uniax"]


def _matcol(name):
    return MAT_PALETTE.get(name, "#888888")


def _matlabel(name):
    return name.replace("medium_density_foam", "foam").replace("_", "-")


def _sig(layup):
    return tuple((p[0], round(float(p[2]))) for p in layup)         # (material, angle) per ply; thickness ignored


def _label(layup):
    mats = [p[0] for p in layup]
    dom = max(layup, key=lambda p: float(p[1]))[0]
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
                reg[sg] = len(labels); labels.append("layup %d" % (len(labels) + 1))  # generic laminate id
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


def _solid_materials(solid_yaml, shell_yaml):
    """per-cell material name of the 2-D solid: each domain's E1 matched to the shell's named materials."""
    sg = read_solid_yaml(solid_yaml)
    mp = np.asarray(sg["material_param"]); dom = np.asarray(sg["cell_domain_ids"])
    shmat = {m["name"]: float(m["elastic"]["E"][0]) for m in yaml.safe_load(open(shell_yaml))["materials"]}
    dom2name = {d: min(shmat, key=lambda k: abs(shmat[k] - mp[d, 0])) for d in range(mp.shape[0])}
    return sg, [dom2name[d] for d in dom]


def _regions(elems, sub):
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
    for i in set(int(v) for v in lay[lay >= 0]):
        for comp in _regions(elems, np.where(lay == i)[0]):
            if len(comp) < 2:
                continue
            mids = emid[comp]; c = mids.mean(0)
            rep = comp[int(np.argmin(np.hypot(*(mids - c).T)))]
            x, y = emid[rep]; e2, e3 = oris[rep, 3:5], oris[rep, 6:8]
            ax.quiver(x, y, e3[0]*L, e3[1]*L, angles="xy", scale_units="xy", scale=1, color="black", width=0.004, zorder=7)
            ax.quiver(x, y, e2[0]*L, e2[1]*L, angles="xy", scale_units="xy", scale=1, color="tab:blue", width=0.004, zorder=6)


def _layup_handles(used, labels):
    from matplotlib.patches import Patch
    return [Patch(color=PALETTE[i % len(PALETTE)], label=labels[i]) for i in sorted(used)]


def _mat_handles(present):
    from matplotlib.patches import Patch
    order = [m for m in MAT_ORDER if m in present] + [m for m in present if m not in MAT_ORDER]
    return [Patch(color=_matcol(m), label=_matlabel(m)) for m in order]


def plot_layup_section(shell_yaml, solid_yaml, reg, out_png, arrow=0.22):
    """2-D solid mesh coloured by MATERIAL (ply layers visible) + 1-D shell LINE mesh coloured by layup,
    with one e2/e3 arrow per connected region."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    sig2idx, labels = reg
    nd, elems, lay, oris, emid = _station(shell_yaml, sig2idx)
    sg, smat = _solid_materials(solid_yaml, shell_yaml)
    P, C = np.asarray(sg["points"])[:, :2], sg["cells"]
    fig, (axS, axL) = plt.subplots(2, 1, figsize=(12.5, 6.4))
    axS.add_collection(PolyCollection([P[c] for c in C], facecolors=[_matcol(m) for m in smat], edgecolors="none"))
    axS.autoscale_view(); axS.set_aspect("equal"); axS.set_title("2-D solid mesh (%d elements)" % len(C))
    axS.set_ylabel("y3 (m)"); axS.set_xticklabels([])
    axS.legend(handles=_mat_handles(set(smat)), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5, title="material")
    axL.add_collection(LineCollection([nd[e[:2]] for e in elems], colors=[PALETTE[i % len(PALETTE)] for i in lay], linewidths=2.0))
    _region_arrows(axL, elems, lay, oris, emid, arrow)
    axL.autoscale_view(); axL.set_aspect("equal")
    axL.set_title("1-D shell line mesh (%d elements)" % len(elems))
    axL.set_xlabel("y2 (m)"); axL.set_ylabel("y3 (m)")
    axL.legend(handles=_layup_handles(set(int(v) for v in lay[lay >= 0]), labels),
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5, title="layup")
    fig.tight_layout(rect=[0, 0, 0.86, 1.0]); fig.savefig(out_png, dpi=145, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_orientation_montage(shell_yamls, rRs, reg, out_png, arrow=0.12, ncol=2):
    """All span stations in ONE image: line cross-section coloured by layup + one e2/e3 arrow per region."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    sig2idx, labels = reg
    n = len(shell_yamls); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel(); used = set()
    for ax, y, r in zip(axes, shell_yamls, rRs):
        nd, elems, lay, oris, emid = _station(y, sig2idx); used |= set(int(v) for v in lay[lay >= 0])
        ax.add_collection(LineCollection([nd[e[:2]] for e in elems], colors=[PALETTE[i % len(PALETTE)] for i in lay], linewidths=1.3))
        _region_arrows(ax, elems, lay, oris, emid, arrow)
        ax.autoscale_view(); ax.set_aspect("equal"); ax.set_title("r = %.1f" % r, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.legend(handles=_layup_handles(used, labels), loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=9, title="layup")
    fig.tight_layout(rect=[0, 0, 0.9, 1.0]); fig.savefig(out_png, dpi=135, bbox_inches="tight"); plt.close(fig)
    return out_png


def plot_span_loft(shell_yamls, rs, reg, out_png):
    """Isometric loft: line cross-sections at their span station ``r`` along the beam axis (x), coloured by
    layup, with a red dot at each section centre and a dotted black beam reference line through them.
    Only the spanwise (r) axis is drawn."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    sig2idx, labels = reg
    fig = plt.figure(figsize=(13, 5.8)); ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    used, centers = set(), []
    for y, r in zip(shell_yamls, rs):
        nd, elems, lay, oris, emid = _station(y, sig2idx); used |= set(int(v) for v in lay[lay >= 0])
        segs = [[(r, nd[e[0], 0], nd[e[0], 1]), (r, nd[e[1], 0], nd[e[1], 1])] for e in elems]
        ax.add_collection3d(Line3DCollection(segs, colors=[PALETTE[i % len(PALETTE)] for i in lay], linewidths=1.5))
        c = nd.mean(0); centers.append((r, c[0], c[1]))
    centers = np.array(centers)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="black", ls=":", lw=1.6, zorder=10)        # beam ref
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="red", s=26, zorder=11)                 # station origins
    ax.set_xlim(min(rs) - 0.05, max(rs) + 0.05); ax.set_ylim(0, 5); ax.set_zlim(-0.9, 0.9)
    ax.set_box_aspect((6.5, 5.0, 1.6))
    ax.view_init(elev=18, azim=-74)
    # spanwise (r) axis only: hide chord (y) and thickness (z)
    ax.set_yticks([0, 2, 4]); ax.set_zticks([-0.5, 0.0, 0.5])   # few, well-spaced ticks (y3 was crowded)
    ax.set_xlabel("r  (span station)", labelpad=14)        # span axis, padded clear of the r tick labels
    ax.set_ylabel("y2 (m)", labelpad=8); ax.set_zlabel("y3 (m)", labelpad=4)   # chord / thickness
    ax.grid(False)
    try:                                                   # fade the panes; keep the three labelled axes + ticks
        ax.xaxis.set_pane_color((1, 1, 1, 0)); ax.yaxis.set_pane_color((1, 1, 1, 0)); ax.zaxis.set_pane_color((1, 1, 1, 0))
    except Exception:
        pass
    fig.legend(handles=_layup_handles(used, labels), loc="center right", fontsize=9, title="layup")  # layups only
    fig.savefig(out_png, dpi=145, bbox_inches="tight"); plt.close(fig)
    return out_png
