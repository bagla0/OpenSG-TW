"""segment_plate.py -- the THROUGH-THICKNESS 1-D structure gene of a plate/shell wall.

A laminate layup is all the geometry the MSG-RM plate law needs: the structure gene is a
1-D mesh through the wall thickness, one (or ``n_per_layer``) higher-order element per ply.
This module turns a layup dictionary into that mesh, writes it as an OpenSG 1-D SG YAML,
reads it back, and homogenizes straight from the file:

    layup dict --plate_sg_yaml--> 1-D SG YAML --rm_plate_from_yaml--> 8x8 ABDG

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

from .msg_rm_plate import _node_grid, rm_plate_msg
from .msg_transverse_shear import plate_8x8


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


def rm_plate_from_yaml(path):
    """Homogenize straight from a plate 1-D SG YAML.

    Returns the ``rm_plate_msg`` result dict with the assembled 8x8 plate law added as
    ``ABDG`` (rows/cols e11,e22,g12,k11,k22,k12,2g13,2g23).
    """
    sg = read_plate_sg_yaml(path)
    r = rm_plate_msg(sg["thick"], sg["angles"], sg["mat_names"], sg["material_db"],
                     n_per_layer=sg["n_per_layer"], elem_order=sg["elem_order"],
                     fraction=sg["fraction"])
    if r["G_msg"] is None:
        raise ValueError("%s: the fitted shear compliance is not SPD" % path)
    r["ABDG"] = np.asarray(plate_8x8(np.asarray(r["A6"]), np.asarray(r["G_msg"])))
    return r
