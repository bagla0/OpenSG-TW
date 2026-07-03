"""
extract_boundaries_dolfinx.py    [ run in the WSL dolfinx environment ]
========================================================================
STAGE 2 of the hybrid MITC-RM tapered-segment pipeline.

    stage 1  make_cylinder_segment.py   (Windows)  -> surface-quad YAML
    stage 2  extract_boundaries_dolfinx.py (WSL)   -> boundary rings + maps   <-- THIS FILE
    stage 3  solve_segment_jax.py        (Windows)  -> RM/MITC 6x6

WHAT dolfinx DOES HERE (and why we use it)
------------------------------------------
The 3D-SG tapered-segment method needs the two END cross-sections of the
segment as their own, lower-dimensional meshes ("boundary SGs").  For a
STRUCTURED prismatic tube we could index those rings by hand, but the whole
point is a pipeline that also works on UNSTRUCTURED, TAPERED blade segments.
dolfinx.mesh.create_submesh does exactly that generically: given the facets
lying on x = x_min (or x_max), it builds a standalone 1-D mesh of that ring
AND returns the maps back to the parent segment (vertex_map / geom_map /
entity_map).  We also carry the per-element material frame (e1,e2,e3) and
subdomain id down onto each boundary the same way OpenSG's ShellSegmentMesh
does (copy the DG0 triad from the parent cell that owns the boundary facet).

This mirrors opensg/mesh/segment.py::ShellSegmentMesh._build_boundary_submeshdata
but is kept self-contained and in ONE coordinate convention:
    beam axis = x ;  cross-section in (y, z) ;  nodes/elements 1-indexed in YAML.

OUTPUT (a single .npz the JAX stage consumes) contains, in dolfinx ordering:
    seg_x (Nn,3), seg_cells (Ne,4), seg_subdom (Ne,), seg_e1/e2/e3 (Ne,3)
    L_x, L_cells, L_subdom, L_e1/e2/e3          (left ring, 1-D line cells)
    R_x, R_cells, R_subdom, R_e1/e2/e3          (right ring)
    L_node2seg (nL,), R_node2seg (nR,)          boundary node -> segment node
    materials (json), layup (json)
so the JAX solver can (a) rebuild the exact same segment mesh, (b) solve each
ring as a 1-D RM/MITC SG, and (c) scatter the ring warping onto the matching
segment boundary DOFs.
"""

import os
import sys
import json
import numpy as np

from mpi4py import MPI
import dolfinx
import basix
from dolfinx.io import gmshio
import gmsh

# The FEniCS YAML uses the C loader; fall back to pure-python if unavailable.
import yaml
try:
    _Loader = yaml.CLoader
except AttributeError:
    _Loader = yaml.SafeLoader


# --------------------------------------------------------------------------- I/O
def load_segment_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=_Loader)


def write_gmsh(seg, msh_path):
    """Write the surface-quad segment as a GMSH 2.2 ASCII mesh.

    We keep our own convention (beam axis = x already), so nodes are written
    verbatim -- NO (node[2],node[0],node[1]) reshuffle that OpenSG uses.
    Element physical/geometric tags carry the layup/subdomain index so the
    material set survives the round-trip through gmsh.
    """
    nodes = seg["nodes"]
    elements = seg["elements"]           # 1-indexed quads [n1 n2 n3 n4]

    # subdomain (layup) id per element, from the element sets
    # (labels may be 0- or 1-indexed; a 0 label can only be 0-indexed)
    subdom = np.zeros(len(elements), dtype=int)
    labs_min = min(min(es["labels"]) for es in seg["sets"]["element"] if es["labels"])
    off = 0 if labs_min == 0 else 1
    for i, es in enumerate(seg["sets"]["element"]):
        for lab in es["labels"]:
            subdom[lab - off] = i

    with open(msh_path, "w") as m:
        m.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
        m.write(f"{len(nodes)}\n")
        for i, nd in enumerate(nodes):
            m.write(f"{i + 1} {nd[0]} {nd[1]} {nd[2]}\n")   # axis=x, no reshuffle
        m.write("$EndNodes\n$Elements\n")
        m.write(f"{len(elements)}\n")
        for i, el in enumerate(elements):
            etype = "3" if len(el) == 4 else "2"            # 3 = quad, 2 = tri
            tag = subdom[i] + 1                              # gmsh tags are 1-based
            conn = " ".join(str(n) for n in el)             # already 1-indexed
            m.write(f"{i + 1} {etype} 2 {tag} {tag} {conn}\n")
        m.write("$EndElements\n")
    return subdom


def build_mesh(msh_path):
    """Read the .msh into dolfinx. Returns (mesh, cell_tags, original_cell_index).

    gmshio permutes cells into dolfinx order; `original_cell_index` lets us map
    our per-element data (orientation) back onto the permuted cells.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    mesh, cell_tags, _ = gmshio.read_from_msh(msh_path, MPI.COMM_WORLD, 0, gdim=3)
    # NB: read_from_msh manages the gmsh session for a file path; do not finalize here.
    original_cell_index = mesh.topology.original_cell_index
    return mesh, cell_tags, original_cell_index


def build_frame(mesh, element_orientations, original_cell_index):
    """DG0 material triad (e1,e2,e3) on the segment, permuted to dolfinx cells.

    element_orientations[i] = [e1(3), e2(3), e3(3)] in OUR axis=x convention,
    so we store the components directly (no OpenSG (o[2],o[0],o[1]) reshuffle).
    """
    VV = dolfinx.fem.functionspace(
        mesh, basix.ufl.element("DG", mesh.topology.cell_name(), 0, shape=(3,)))
    e1 = dolfinx.fem.Function(VV)
    e2 = dolfinx.fem.Function(VV)
    e3 = dolfinx.fem.Function(VV)
    for k, ii in enumerate(original_cell_index):      # k = dolfinx cell, ii = our cell
        o = element_orientations[ii]
        e1.x.array[3 * k:3 * k + 3] = o[0:3]
        e2.x.array[3 * k:3 * k + 3] = o[3:6]
        e3.x.array[3 * k:3 * k + 3] = o[6:9]
    return e1, e2, e3


# ----------------------------------------------------------------- boundary rings
def extract_ring(mesh, cell_subdom, e1, e2, e3, x_target=None, tol=1e-6, facets=None):
    """Extract one end cross-section (a 1-D ring).

    Follows ShellSegmentMesh._build_boundary_submeshdata:
      1. select the cross-section facets -- either the given `facets` array
         (robust path: exterior facets partitioned by end), or the plane
         x == x_target (legacy prismatic path),
      2. create_submesh -> standalone 1-D ring + maps to the parent,
      3. copy the material frame + subdomain from the PARENT cell that owns
         each ring facet (on the cross-section the axial component of the
         in-plane frame is dropped: e1_ring = axial normal = (1,0,0)).

    Returns a dict with ring geometry/connectivity/frame/subdomain and the
    boundary-node -> segment-node index map (geom_map).
    """
    tdim = mesh.topology.dim          # 2 (surface)
    fdim = tdim - 1                   # 1 (edges = cross-section facets)

    if facets is None:                # legacy plane selection (prismatic meshes)
        def on_plane(x):
            return np.isclose(x[0], x_target, atol=tol)
        facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, on_plane)
    facets = np.asarray(facets, dtype=np.int32)
    ring, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(mesh, fdim, facets)

    # parent facet -> parent cell (to inherit material data onto the ring)
    mesh.topology.create_connectivity(fdim, tdim)
    f2c = mesh.topology.connectivity(fdim, tdim)

    nrc = ring.topology.index_map(ring.topology.dim).size_local   # ring cell count
    r_sub = np.zeros(nrc, dtype=np.int32)
    r_e1 = np.zeros((nrc, 3)); r_e2 = np.zeros((nrc, 3)); r_e3 = np.zeros((nrc, 3))
    for i in range(nrc):
        parent_facet = entity_map[i]                 # ring cell i -> parent facet
        parent_cell = f2c.links(parent_facet)[0]     # boundary facet has one owner cell
        r_sub[i] = cell_subdom[parent_cell]
        # e1 on the cross-section is the axial normal; keep only the in-plane e2,e3
        r_e1[i] = (1.0, 0.0, 0.0)
        r_e2[i] = (0.0, e2.x.array[3 * parent_cell + 1], e2.x.array[3 * parent_cell + 2])
        r_e3[i] = (0.0, e3.x.array[3 * parent_cell + 1], e3.x.array[3 * parent_cell + 2])

    # ring cell -> (local vertex ids); map to parent-segment node ids via geom_map
    ring.topology.create_connectivity(ring.topology.dim, 0)
    r_c2v = ring.topology.connectivity(ring.topology.dim, 0)
    r_cells_local = np.array([r_c2v.links(i) for i in range(nrc)], dtype=np.int64)

    return {
        "x": np.asarray(ring.geometry.x, dtype=float),   # ring node coords
        "cells": r_cells_local,                          # (nrc, 2) local line cells
        "subdom": r_sub,
        "e1": r_e1, "e2": r_e2, "e3": r_e3,
        "node2seg": np.asarray(geom_map, dtype=np.int64),  # ring node -> segment node
    }


# ------------------------------------------------------------------------- driver
def extract(seg_yaml, out_npz, work_msh="SG_mesh.msh"):
    seg = load_segment_yaml(seg_yaml)
    subdom = write_gmsh(seg, work_msh)
    mesh, cell_tags, oci = build_mesh(work_msh)
    e1, e2, e3 = build_frame(mesh, seg["elementOrientations"], oci)

    # segment cell subdomain in dolfinx order (cell_tags carries the gmsh tag-1)
    cell_subdom = np.zeros(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
    cell_subdom[cell_tags.indices] = cell_tags.values - 1

    xg = mesh.geometry.x
    x_min, x_max = float(xg[:, 0].min()), float(xg[:, 0].max())
    # ROBUST end selection (twisted blade end faces are NOT planar): take ALL
    # exterior facets (the dolfinx facet->one-cell criterion, exactly what
    # ShellSegmentMesh uses) and partition them by which end their midpoint is
    # nearer.  No plane tolerance involved.
    tdim = mesh.topology.dim; fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    ext = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    mid = dolfinx.mesh.compute_midpoints(mesh, fdim, np.asarray(ext, dtype=np.int32))
    xc = 0.5 * (x_min + x_max)
    left = extract_ring(mesh, cell_subdom, e1, e2, e3, facets=ext[mid[:, 0] < xc])
    right = extract_ring(mesh, cell_subdom, e1, e2, e3, facets=ext[mid[:, 0] >= xc])

    # segment connectivity in dolfinx order (Ne, 4)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    c2v = mesh.topology.connectivity(mesh.topology.dim, 0)
    ncell = mesh.topology.index_map(mesh.topology.dim).size_local
    seg_cells = np.array([c2v.links(i) for i in range(ncell)], dtype=np.int64)

    np.savez(
        out_npz,
        seg_x=np.asarray(xg, dtype=float), seg_cells=seg_cells, seg_subdom=cell_subdom,
        seg_e1=e1.x.array.reshape(-1, 3), seg_e2=e2.x.array.reshape(-1, 3),
        seg_e3=e3.x.array.reshape(-1, 3),
        L_x=left["x"], L_cells=left["cells"], L_subdom=left["subdom"],
        L_e1=left["e1"], L_e2=left["e2"], L_e3=left["e3"], L_node2seg=left["node2seg"],
        R_x=right["x"], R_cells=right["cells"], R_subdom=right["subdom"],
        R_e1=right["e1"], R_e2=right["e2"], R_e3=right["e3"], R_node2seg=right["node2seg"],
        materials=json.dumps(seg["materials"]),
        sections=json.dumps(seg["sections"]),
        x_min=x_min, x_max=x_max,
    )
    print("segment: %d nodes, %d quads | left ring: %d nodes, %d cells | right ring: %d nodes"
          % (xg.shape[0], ncell, left["x"].shape[0], left["cells"].shape[0], right["x"].shape[0]))
    print("wrote", out_npz)


if __name__ == "__main__":
    # usage: python extract_boundaries_dolfinx.py meshes/seg_iso_hR0.1.yaml out/seg_iso_hR0.1.npz
    seg_yaml = sys.argv[1] if len(sys.argv) > 1 else "meshes/seg_iso_hR0.1.yaml"
    out_npz = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(seg_yaml))[0] + ".npz"
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    extract(seg_yaml, out_npz)
