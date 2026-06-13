"""
MSG Shell — Cross-Section Mesh Utilities and YAML Loader

Provides:
  - load_yaml         : parse OpenSG-format YAML cross-section files
  - order_mesh        : chain disordered line elements into CCW closed loop
                        and insert midside nodes for 3-node quadratic elements
  - compute_curvature : per-element k22 via 3-point circumscribed circle
"""

import numpy as np
import yaml


# =============================================================================
# YAML Loader (OpenSG format)
# =============================================================================

def load_yaml(yaml_path):
    """Load an OpenSG-format YAML cross-section file.

    OpenSG uses space-separated YAML lists without commas, e.g. ``[1 2]``
    rather than ``[1, 2]``.  ``yaml.safe_load`` returns these as a single
    string inside a list, so a ``_parse_row`` helper is applied.

    Parameters
    ----------
    yaml_path : str — path to .yaml file

    Returns
    -------
    nodes_3d     : (N, 3) ndarray — node coordinates (z=0 for cross-sections)
    elements     : list of [n1, n2] — 1-based line element connectivity
    material_db  : dict {name: {E, G, nu, rho}}
    layup_db     : dict {layup_name: {mat_names, thick, angles}}
    elem_to_layup: dict {elem_id_1based: layup_name}
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    def _parse_row(row):
        if isinstance(row, str):
            return row.strip('[]').split()
        if isinstance(row, (list, tuple)):
            if len(row) == 1 and isinstance(row[0], str):
                return row[0].strip('[]').split()
            return [str(v) for v in row]
        return [str(row)]

    nodes_3d = np.array([[float(v) for v in _parse_row(nd)]
                         for nd in data['nodes']])
    elements = [[int(v) for v in _parse_row(el)]
                for el in data['elements']]

    material_db = {}
    for mat in data['materials']:
        material_db[mat['name']] = {
            'E':   mat['elastic']['E'],
            'G':   mat['elastic']['G'],
            'nu':  mat['elastic']['nu'],
            'rho': mat.get('density', 0.0),
        }

    elem_sets = {s['name']: s['labels'] for s in data['sets']['element']}

    layup_db = {}
    elem_to_layup = {}
    for sec in data['sections']:
        ln = sec['elementSet']
        layers = sec['layup']
        layup_db[ln] = {
            'mat_names': [ly[0] for ly in layers],
            'thick':     [ly[1] for ly in layers],
            'angles':    [ly[2] for ly in layers],
        }
        for eid in elem_sets[ln]:
            elem_to_layup[eid] = ln

    return nodes_3d, elements, material_db, layup_db, elem_to_layup


# =============================================================================
# Mesh Ordering
# =============================================================================

def order_mesh(nodes_3d, elements_1b, elem_to_layup):
    """Chain disordered line elements into a continuous CCW closed loop.

    Walks the adjacency graph from the first element's start node.
    If the resulting polygon has CW orientation (area < 0), the order is
    reversed.  Midside nodes are inserted at chord midpoints to form
    3-node quadratic elements.

    Parameters
    ----------
    nodes_3d     : (N, 3) array — 3D node coordinates (x=y2, y=y3 used)
    elements_1b  : list of [n1, n2] — 1-based connectivity
    elem_to_layup: dict {elem_id_1based: layup_name}

    Returns
    -------
    nodes_2d      : (2*M+1, 2) — ordered [y2, y3] coords with midside nodes
    cells         : (M, 3) int64 — element connectivity (0-based, 3-node)
    layup_per_elem: list[str] — layup name for each element (traversal order)
    is_closed     : bool
    """
    elems = [(e[0] - 1, e[1] - 1) for e in elements_1b]
    n_elems = len(elems)

    adj = {}
    for idx, (n1, n2) in enumerate(elems):
        adj.setdefault(n1, []).append((n2, idx))
        adj.setdefault(n2, []).append((n1, idx))

    start = elems[0][0]
    visited = set()
    nids = [start]
    eids = []
    cur = start

    while len(visited) < n_elems:
        found = False
        for nb, eidx in adj[cur]:
            if eidx not in visited:
                visited.add(eidx)
                nids.append(nb)
                eids.append(eidx)
                cur = nb
                found = True
                break
        if not found:
            break

    is_closed = (nids[-1] == nids[0])

    if len(visited) < n_elems:
        print(f"  WARNING: only {len(visited)}/{n_elems} elements chained "
              f"(mesh may be disconnected)")

    corner_coords = nodes_3d[nids, :2]

    # Signed area — ensure CCW
    area = 0.0
    for i in range(len(corner_coords) - 1):
        area += (corner_coords[i, 0] * corner_coords[i + 1, 1]
                 - corner_coords[i + 1, 0] * corner_coords[i, 1])
    area *= 0.5

    if area < 0:
        nids = nids[::-1]
        eids = eids[::-1]
        corner_coords = nodes_3d[nids, :2]

    n_elem = len(nids) - 1
    mid_coords = 0.5 * (corner_coords[:-1] + corner_coords[1:])

    nodes_2d = np.zeros((2 * n_elem + 1, 2))
    nodes_2d[0::2] = corner_coords
    nodes_2d[1::2] = mid_coords

    cells = np.column_stack([
        np.arange(0, 2 * n_elem, 2),
        np.arange(1, 2 * n_elem + 1, 2),
        np.arange(2, 2 * n_elem + 2, 2),
    ]).astype(np.int64)

    layup_per_elem = [elem_to_layup[eidx + 1] for eidx in eids]

    return nodes_2d, cells, layup_per_elem, is_closed


# =============================================================================
# Curvature Computation
# =============================================================================

def compute_curvature(nodes_2d, cells, is_closed=True):
    """Per-element curvature k22 via 3-point circumscribed circle.

    Uses element midside nodes (``cells[:, 1]``) as element centres.

    Sign convention: k22 = -(signed curvature), so k22 = -1/R for a CCW
    circle of radius R.

    Parameters
    ----------
    nodes_2d  : (2*N+1, 2) node coordinates including midside nodes
    cells     : (N, 3) element connectivity
    is_closed : bool — wrap neighbour lookup for a closed loop

    Returns
    -------
    k22 : (N,) ndarray
    """
    n_elem = cells.shape[0]
    mid = nodes_2d[cells[:, 1]]

    k22 = np.zeros(n_elem)
    for i in range(n_elem):
        im = (i - 1) % n_elem if is_closed else max(i - 1, 0)
        ip = (i + 1) % n_elem if is_closed else min(i + 1, n_elem - 1)

        d12 = mid[i]  - mid[im]
        d23 = mid[ip] - mid[i]

        cross = d12[0] * d23[1] - d12[1] * d23[0]
        l12   = np.linalg.norm(d12)
        l23   = np.linalg.norm(d23)
        l31   = np.linalg.norm(mid[ip] - mid[im])
        denom = l12 * l23 * l31

        k22[i] = -2.0 * cross / denom if denom > 1e-30 else 0.0

    return k22
