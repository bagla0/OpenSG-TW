"""segment_plate.py -- MESH-GENERATION HELPER for the through-thickness 1-D plate SG.

A laminate layup is all the geometry the MSG-RM plate law needs: the structure gene is a
1-D mesh through the wall thickness, one (or ``n_per_layer``) higher-order element per ply.
This module ONLY generates and reads that mesh -- it does no homogenization; feed the
result to ``msg_rm_plate.rm_plate_msg``:

    layup dict --plate_sg_yaml--> 1-D SG YAML --read_plate_sg_yaml--> rm_plate_msg(...)

Distinguish this from the 1-D SHELL SG YAML (e.g. ``examples/data/1d_yaml/st15_shell.yaml``):
that one is the CONTOUR of a cross-section -- line elements running around the airfoil, each
carrying a layup.  The file written here is the wall itself, discretised THROUGH the
thickness; its nodes are x3 coordinates, not points in the cross-section plane.

Mesh convention (identical to what ``msg_rm_plate`` builds internally, asserted in
``tests/test_segment_plate.py``):
  * ``elem_order = 4`` by default -> 5-NODED QUARTIC elements, the degree that represents
    the whole warping ladder exactly per ply (V0 quadratic, V1 cubic, V2 quartic)
  * ``n_per_layer = 1`` by default -> one element per ply
  * nodes are equispaced inside each element, elements share their end nodes, so
    n_node = elem_order * n_elem + 1
  * x3 is measured from the REFERENCE PLANE set by ``fraction`` (0 = bottom/OML face,
    0.5 = mid-surface, 1 = top/IML), so the nodes run from -fraction*h to (1-fraction)*h

YAML layout::

    sg:        type/elem_order/n_per_layer/reference_fraction/thickness
    nodes:     [x3] per node, bottom to top
    elements:  1-based connectivity, (elem_order + 1) nodes each
    materials: name + elastic {E, G, nu} + density   (same block as the shell YAML)
    sets:      element sets, one per PLY (ply_1, ply_2, ...)
    sections:  per ply set: material, angle, thickness
"""
import os

import numpy as np
import yaml

from .msg_rm_plate import _node_grid


def plate_sg_mesh(thick, n_per_layer=1, elem_order=4, fraction=0.5):
    """The through-thickness 1-D mesh for a layup.

    Returns ``(node_x, elements, elem_ply)``:
      node_x    (n_node,)            x3 of each node, measured from the reference plane
      elements  (n_elem, p+1) int    0-based connectivity
      elem_ply  (n_elem,) int        which ply each element belongs to
    """
    thick = [float(t) for t in thick]
    p = int(elem_order)
    n_elem = len(thick) * int(n_per_layer)
    node_x = _node_grid(thick, int(n_per_layer), p, float(fraction) * sum(thick))
    elements = np.stack([np.arange(p * e, p * e + p + 1) for e in range(n_elem)])
    elem_ply = np.repeat(np.arange(len(thick)), int(n_per_layer))
    return node_x, elements, elem_ply


def plate_sg_dict(layup, material_db, n_per_layer=1, elem_order=4, fraction=0.5):
    """Build the 1-D plate SG as a plain dict (the YAML document, unwritten).

    ``layup`` is the layup dictionary used everywhere else in OpenSG::

        {"mat_names": [...], "thick": [...], "angles": [...]}      bottom ply first
    """
    mats = [str(m) for m in layup["mat_names"]]
    thick = [float(t) for t in layup["thick"]]
    angles = [float(a) for a in layup["angles"]]
    if not (len(mats) == len(thick) == len(angles)):
        raise ValueError("layup lists disagree: %d materials, %d thicknesses, %d angles"
                         % (len(mats), len(thick), len(angles)))
    if min(thick) <= 0:
        raise ValueError("ply thicknesses must be positive, got %s" % thick)
    missing = [m for m in mats if m not in material_db]
    if missing:
        raise KeyError("materials not in the database: %s" % ", ".join(sorted(set(missing))))

    node_x, elements, elem_ply = plate_sg_mesh(thick, n_per_layer, elem_order, fraction)
    n_per_layer = int(n_per_layer)

    sets, sections = [], []
    for k in range(len(thick)):
        labels = [int(e) + 1 for e in np.flatnonzero(elem_ply == k)]     # 1-based
        name = "ply_%d" % (k + 1)
        sets.append({"name": name, "labels": labels})
        sections.append({"elementSet": name, "material": mats[k],
                         "angle": angles[k], "thickness": thick[k]})

    used = [m for m in dict.fromkeys(mats)]
    materials = [{"name": m, "density": float(material_db[m].get("rho", 0.0)),
                  "elastic": {"E": [float(v) for v in material_db[m]["E"]],
                              "G": [float(v) for v in material_db[m]["G"]],
                              "nu": [float(v) for v in material_db[m]["nu"]]}}
                 for m in used]

    return {"sg": {"type": "plate_1d", "elem_order": int(elem_order),
                   "n_per_layer": n_per_layer, "reference_fraction": float(fraction),
                   "thickness": float(sum(thick)), "n_ply": len(thick)},
            "nodes": [[float(x)] for x in node_x],
            "elements": [[int(n) + 1 for n in row] for row in elements],
            "materials": materials,
            "sets": {"element": sets},
            "sections": sections}


def plate_sg_yaml(path, layup, material_db, n_per_layer=1, elem_order=4, fraction=0.5):
    """Write the through-thickness 1-D SG YAML for a layup.  Returns the written dict."""
    doc = plate_sg_dict(layup, material_db, n_per_layer, elem_order, fraction)
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
    return doc


def read_plate_sg_yaml(path, atol=1e-9):
    """Read a plate 1-D SG YAML back into the arguments ``rm_plate_msg`` takes.

    Returns a dict with keys thick / angles / mat_names / material_db / fraction /
    n_per_layer / elem_order / node_x.

    Ply thicknesses and the reference fraction are taken from the stored fields and then
    CROSS-CHECKED against the mesh (summed element spans, node_x[0]) to ``atol`` relative.
    Re-deriving them from the mesh instead costs exactness -- summing node differences
    turns a 0.004 ply into 0.004000000000000002, which propagates to ~1e-14 in the 8x8 --
    while checking keeps the round trip bit-exact AND still catches a corrupt or
    hand-edited file, where mesh and header would genuinely disagree.
    """
    with open(path, "r") as f:
        doc = yaml.safe_load(f)
    sg = doc.get("sg", {})
    if sg.get("type") not in (None, "plate_1d"):
        raise ValueError("%s is a %r SG, not a through-thickness plate_1d SG"
                         % (path, sg.get("type")))

    node_x = np.array([float(np.atleast_1d(nd)[0]) for nd in doc["nodes"]])
    elements = [[int(v) - 1 for v in el] for el in doc["elements"]]
    material_db = {m["name"]: {"E": [float(v) for v in m["elastic"]["E"]],
                               "G": [float(v) for v in m["elastic"]["G"]],
                               "nu": [float(v) for v in m["elastic"]["nu"]],
                               "rho": float(m.get("density", 0.0))}
                   for m in doc["materials"]}
    elem_sets = {s["name"]: [int(v) - 1 for v in s["labels"]] for s in doc["sets"]["element"]}

    thick, angles, mat_names, counts = [], [], [], []
    for sec in doc["sections"]:
        eids = elem_sets[sec["elementSet"]]
        span = sum(abs(node_x[elements[e][-1]] - node_x[elements[e][0]]) for e in eids)
        t = float(sec["thickness"]) if "thickness" in sec else float(span)
        if abs(t - span) > atol * max(span, 1e-300):
            raise ValueError("%s: ply %r declares thickness %g but its elements span %g"
                             % (path, sec["elementSet"], t, span))
        thick.append(t)
        angles.append(float(sec["angle"]))
        mat_names.append(str(sec["material"]))
        counts.append(len(eids))
    if len(set(counts)) != 1:
        raise ValueError("%s: plies carry different element counts %s" % (path, counts))

    h = float(sum(thick))
    frac = float(sg["reference_fraction"]) if "reference_fraction" in sg \
        else (float(-node_x[0] / h) if h > 0 else 0.0)
    if h > 0 and abs(-frac * h - node_x[0]) > atol * h:
        raise ValueError("%s: reference_fraction %g puts the bottom face at %g, "
                         "but the first node is at %g" % (path, frac, -frac * h, node_x[0]))
    return {"thick": thick, "angles": angles, "mat_names": mat_names,
            "material_db": material_db, "n_per_layer": counts[0],
            "elem_order": len(elements[0]) - 1, "fraction": frac, "node_x": node_x}


def plot_plate_sg(path, png_path=None):
    """Render the ACTUAL mesh in a plate 1-D SG YAML (nodes and connectivity read from
    the file, never a sketch).  Left: the through-thickness discretisation, plies as
    material-coloured bands with every node marked.  Right: the ply material frames,
    e1 (fibre) red, e2 blue, e3 black -- e3 is out of plane, drawn as a circled dot.

    Returns the PNG path (default: the YAML path with a .png suffix).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with open(path, "r") as f:
        doc = yaml.safe_load(f)
    sg = read_plate_sg_yaml(path)
    node_x = sg["node_x"]
    elements = [[int(v) - 1 for v in el] for el in doc["elements"]]
    png_path = png_path or (os.path.splitext(path)[0] + ".png")

    h = float(sum(sg["thick"]))
    bot = np.concatenate([[0.0], np.cumsum(sg["thick"])]) - sg["fraction"] * h
    mats = list(dict.fromkeys(sg["mat_names"]))
    cmap = plt.get_cmap("tab10")
    colr = {m: cmap(i % 10) for i, m in enumerate(mats)}

    fig, (axm, axo) = plt.subplots(1, 2, figsize=(9.5, 6.0),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---------------- left: the mesh itself -------------------------------------
    for k in range(len(sg["thick"])):
        axm.add_patch(Rectangle((0.0, bot[k]), 1.0, bot[k + 1] - bot[k],
                                facecolor=colr[sg["mat_names"][k]], alpha=0.30,
                                edgecolor="none"))
        axm.axhline(bot[k + 1], color="0.45", lw=0.9)          # ply interface
        axm.text(0.04, 0.5 * (bot[k] + bot[k + 1]),
                 "ply %d: %s\n%.1f mm / %g$^\\circ$"
                 % (k + 1, sg["mat_names"][k], 1e3 * sg["thick"][k], sg["angles"][k]),
                 va="center", ha="left", fontsize=9)
    axm.axhline(bot[0], color="0.45", lw=0.9)
    for e, el in enumerate(elements):                      # element spans + end nodes
        axm.plot([0.5, 0.5], [node_x[el[0]], node_x[el[-1]]], "-", color="0.35", lw=1.4)
        for nd in (el[0], el[-1]):
            axm.plot(0.5, node_x[nd], "o", ms=7, mfc="w", mec="k", mew=1.4, zorder=3)
    interior = sorted(set(range(len(node_x))) - {el[0] for el in elements}
                      - {el[-1] for el in elements})
    axm.plot([0.5] * len(interior), node_x[interior], "o", ms=4.5, color="k", zorder=3,
             label="interior nodes")
    axm.plot([], [], "o", ms=7, mfc="w", mec="k", mew=1.4, label="element end nodes")
    axm.axhline(0.0, ls="--", lw=1.1, color="crimson")
    axm.text(0.02, 0.0, "reference plane (fraction = %.2f)" % sg["fraction"],
             color="crimson", fontsize=8.5, va="bottom")
    axm.set_xlim(0.0, 1.0); axm.set_ylim(bot[0] - 0.06 * h, bot[-1] + 0.06 * h)
    axm.set_xticks([]); axm.set_ylabel("$x_3$  [m]", fontsize=11)
    axm.legend(fontsize=8.5, loc="lower left", frameon=False)
    axm.set_frame_on(False)

    # ---------------- right: the ply material frames ----------------------------
    # arrows sized to the THINNEST ply so every frame stays inside its own band
    L = 0.35 * min(sg["thick"])
    for k in range(len(sg["thick"])):
        yc = 0.5 * (bot[k] + bot[k + 1])
        th = np.deg2rad(sg["angles"][k])
        axo.add_patch(Rectangle((-0.45 * h, bot[k]), 0.9 * h, bot[k + 1] - bot[k],
                                facecolor=colr[sg["mat_names"][k]], alpha=0.18,
                                edgecolor="none"))
        axo.annotate("", xy=(L * np.cos(th), yc + L * np.sin(th)), xytext=(0, yc),
                     arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8))
        axo.annotate("", xy=(-L * np.sin(th), yc + L * np.cos(th)), xytext=(0, yc),
                     arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6))
        axo.plot(0, yc, "o", ms=9, mfc="w", mec="k", mew=1.5, zorder=3)
        axo.plot(0, yc, ".", ms=3, color="k", zorder=4)
        axo.axhline(bot[k + 1], color="0.45", lw=0.9)
    axo.axhline(bot[0], color="0.45", lw=0.9)
    axo.plot([], [], "-", color="crimson", lw=1.8, label="$e_1$ (fibre)")
    axo.plot([], [], "-", color="tab:blue", lw=1.6, label="$e_2$")
    axo.plot([], [], "o", ms=9, mfc="w", mec="k", mew=1.5, label="$e_3$ (out of plane)")
    axo.set_xlim(-0.45 * h, 0.45 * h); axo.set_ylim(bot[0] - 0.06 * h, bot[-1] + 0.06 * h)
    axo.set_xticks([]); axo.set_yticks([])
    axo.legend(fontsize=8.5, loc="lower left", frameon=False)
    axo.set_frame_on(False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path
