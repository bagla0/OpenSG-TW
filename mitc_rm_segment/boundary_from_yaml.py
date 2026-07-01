"""
boundary_from_yaml.py    [ pure numpy, NO dolfinx -> NO renumbering ]
========================================================================
Reliable, renumber-free boundary extraction for a STRUCTURED surface-quad
segment -- the alternative to the dolfinx `create_submesh` path.

Because we generate the mesh ourselves, we already KNOW which nodes are the two
end cross-sections (x = x_min / x_max).  Slicing them here keeps the ORIGINAL
YAML node ids, so:
   * NO node/cell renumbering ever happens;
   * a ring node IS a segment node  ->  the ring<->segment DOF match is the
     identity (L_node2seg = the ring's own YAML node ids) -- nothing to renumber.

Each ring edge inherits its material frame from its PARENT boundary quad
(e1 = axial, e2/e3 = the quad's in-plane hoop/normal), exactly as the dolfinx
path does, but without the permutation.

ALWAYS emits the e1/e2/e3 orientation precheck PNGs (segment + both rings) and a
numeric frame report, so orientation consistency is verified every run.

Exports the SAME .npz schema as extract_boundaries_dolfinx.py, so stage 3
(solve_segment_jax.py) is path-agnostic.  This file is also the ground-truth the
dolfinx path is cross-checked against (it must renumber back to match).
"""

import os
import sys
import json
import numpy as np
import yaml

from orient_check import frame_report, orientation_png, orientation_png_ring


def load_segment_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def subdomain_ids(seg):
    """layup/subdomain index per element from the element sets (0-based)."""
    ne = len(seg["elements"])
    subdom = np.zeros(ne, dtype=np.int32)
    for i, es in enumerate(seg["sets"]["element"]):
        for lab in es["labels"]:               # labels are 1-indexed element ids
            subdom[lab - 1] = i
    return subdom


def write_boundary_yaml(order, nodes, re2, re3, sections, materials, path):
    """Write one end ring as a standalone 1-D OpenSG cross-section YAML.

    Cross-section stored as (y, z, 0) so the axial x collapses to the dummy 3rd
    coord; the frame is written in that convention (e1 = axial = (0,0,1), e2/e3
    the in-plane hoop/normal).  Node order = `order` (theta-sorted segment node
    ids) so the boundary solution maps back to the segment by node2seg = order.
    """
    m = len(order)
    d = {
        "nodes": [[float(nodes[nd, 1]), float(nodes[nd, 2]), 0.0] for nd in order],
        "elements": [[i + 1, (i + 1) % m + 1] for i in range(m)],           # closed loop
        "sections": sections,
        "sets": {"element": [{"name": sections[0]["elementSet"], "labels": list(range(1, m + 1))}]},
        "materials": materials,
        "elementOrientations": [[0.0, 0.0, 1.0,
                                 float(re2[i][1]), float(re2[i][2]), 0.0,
                                 float(re3[i][1]), float(re3[i][2]), 0.0] for i in range(m)],
    }
    with open(path, "w") as f:
        yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False)
    return path


def extract(seg_yaml, out_npz, tol=1e-6):
    seg = load_segment_yaml(seg_yaml)
    nodes = np.array(seg["nodes"], dtype=float)              # (Nn,3), YAML order
    quads = np.array(seg["elements"], dtype=np.int64) - 1    # (Ne,4), 0-based
    ori = np.array(seg["elementOrientations"], dtype=float)  # (Ne,9)
    e1s, e2s, e3s = ori[:, 0:3], ori[:, 3:6], ori[:, 6:9]
    subdom = subdomain_ids(seg)
    out_dir = os.path.dirname(out_npz) or "."
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(out_npz))[0]

    # --- orientation PRECHECK on the full segment (mandatory byproduct) ----------
    ok, txt = frame_report(nodes, quads, e1s, e2s, e3s)
    print("segment " + txt)
    orientation_png(nodes, quads, e1s, e2s, e3s,
                    os.path.join(out_dir, "orient_%s_segment.png" % tag),
                    title="%s  segment e1/e2/e3" % tag, step=2)

    # node -> incident quads (to find each ring edge's parent boundary quad)
    node_to_quads = [[] for _ in range(len(nodes))]
    for q, quad in enumerate(quads):
        for nd in quad:
            node_to_quads[nd].append(q)

    x = nodes[:, 0]
    x_min, x_max = float(x.min()), float(x.max())
    bundle = dict(seg_x=nodes, seg_cells=quads, seg_subdom=subdom,
                  seg_e1=e1s, seg_e2=e2s, seg_e3=e3s,
                  materials=json.dumps(seg["materials"]),
                  sections=json.dumps(seg["sections"]),
                  x_min=x_min, x_max=x_max)

    for side, xt in [("L", x_min), ("R", x_max)]:
        ring = np.where(np.isclose(x, xt, atol=tol))[0]      # YAML node ids on this end
        th = np.arctan2(nodes[ring, 2], nodes[ring, 1])
        order = ring[np.argsort(th)]                          # ordered CCW by hoop angle
        m = len(order)
        cells = np.array([[i, (i + 1) % m] for i in range(m)], dtype=np.int64)  # closed loop (local)

        # per-edge frame from the parent boundary quad (unique quad holding both nodes)
        re1 = np.zeros((m, 3)); re2 = np.zeros((m, 3)); re3 = np.zeros((m, 3)); rsub = np.zeros(m, np.int32)
        for i in range(m):
            a, bb = order[i], order[(i + 1) % m]
            common = set(node_to_quads[a]) & set(node_to_quads[bb])
            q = min(common)                                   # boundary hoop edge -> exactly one quad
            re1[i] = (1.0, 0.0, 0.0)                           # e1 on the cross-section = axial normal
            re2[i] = (0.0, e2s[q, 1], e2s[q, 2])              # in-plane hoop
            re3[i] = (0.0, e3s[q, 1], e3s[q, 2])              # in-plane inward normal
            rsub[i] = subdom[q]

        ring_x = nodes[order]                                 # ring node coords (ordered)
        okr, txtr = frame_report(ring_x, cells, re1, re2, re3)
        print("%s-ring %s" % (side, txtr))
        orientation_png_ring(ring_x, cells, re2, re3,
                             os.path.join(out_dir, "orient_%s_ring_%s.png" % (tag, side)),
                             title="%s  %s-ring e2/e3" % (tag, side))

        byaml = os.path.join(out_dir, "boundary_%s_%s.yaml" % (tag, side))
        write_boundary_yaml(order, nodes, re2, re3, seg["sections"], seg["materials"], byaml)
        print("  wrote boundary YAML -> %s" % os.path.basename(byaml))

        bundle["%s_x" % side] = ring_x
        bundle["%s_cells" % side] = cells
        bundle["%s_subdom" % side] = rsub
        bundle["%s_e1" % side] = re1
        bundle["%s_e2" % side] = re2
        bundle["%s_e3" % side] = re3
        bundle["%s_node2seg" % side] = order                  # IDENTITY: ring node = YAML segment node
        print("  %s ring: %d nodes at x=%.4f (YAML ids, no renumbering)" % (side, m, xt))

    np.savez(out_npz, **bundle)
    print("wrote", out_npz, "| orientation PNGs in", out_dir)
    return out_npz


if __name__ == "__main__":
    seg_yaml = sys.argv[1] if len(sys.argv) > 1 else "meshes/seg_iso_hR0.1.yaml"
    out_npz = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        "out", os.path.splitext(os.path.basename(seg_yaml))[0] + "_direct.npz")
    extract(seg_yaml, out_npz)
