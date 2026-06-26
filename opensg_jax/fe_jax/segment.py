"""JAX OpenSG-TW: 2D-solid SG YAML reader (a JAX analogue of the FEniCS `opensg/mesh/segment.py`).

Reads the same 2D-solid YAML the FEniCS solid consumes (the `solid_*.yaml` produced from a PreVABS `.sg`)
and returns exactly the data structures the JAX homogenizer (Beam_solid pipeline) needs -- i.e. the same
tuple that `generate_msh_from_sc` produces from a `.sc` file, but sourced from the YAML:

    n_sg, points (V, n_sg), cells (E, N) 0-indexed, cell_domain_ids (E,), elem_rotation (E, 9),
    mat_seq (n_dom,), mat_angles (n_dom,), material_param (n_mats, 9)

YAML schema (OpenSG 2D-solid, space-separated strings inside the flow lists):
  nodes: [["y z x"], ...]                      -- cross-section = (y, z); x = beam axis (0 here)
  elements: [["n1 n2 n3(n4)"], ...]            -- 1-indexed; tri (3) or quad (4)
  sets: {element: [{name, labels:[1-indexed elem ids]}, ...]}
  elementOrientations: [[a11..a33], ...]       -- per-element 3x3 frame (e1,e2,e3), 9 components
  materials: [{name, E:[3], G:[3], nu:[3], rho}, ...]

This file is standalone -- it does NOT modify Beam_solid.py or any existing module.
"""
import numpy as np
import yaml


def _row(item):
    """A node/element YAML item is a 1-element list holding a space-separated string (or a real list)."""
    if isinstance(item, (list, tuple)):
        if len(item) == 1 and isinstance(item[0], str):
            return item[0].split()
        return item
    if isinstance(item, str):
        return item.split()
    return [item]


def read_solid_yaml(path):
    """Parse a 2D-solid SG YAML into the JAX-homogenizer input structures. Returns a dict."""
    d = yaml.safe_load(open(path))

    # --- nodes -> points (cross-section y,z) ; x (third coord) is the beam axis ---
    nodes = np.array([[float(v) for v in _row(n)] for n in d["nodes"]], dtype=np.float64)  # (V, 3) = [y z x]
    n_sg = 2
    points = nodes[:, 0:n_sg].copy()                          # (V, 2) cross-section coordinates

    # --- elements (1-indexed strings) -> 0-indexed connectivity ---
    elem_lists = [[int(v) for v in _row(e)] for e in d["elements"]]
    nN = len(elem_lists[0])
    if any(len(e) != nN for e in elem_lists):
        raise ValueError("mixed element types (tri+quad) not yet supported by this reader")
    cells = np.array(elem_lists, dtype=np.int64) - 1         # (E, N), 0-indexed
    E = cells.shape[0]
    elem_type = {3: "triangle", 4: "quad"}.get(nN)
    if elem_type is None:
        raise ValueError("unsupported element node count %d" % nN)

    # --- element sets -> per-element domain (material/layer) id ---
    materials = d["materials"]
    mat_lookup = {m["name"]: m for m in materials}
    cell_domain_ids = np.zeros(E, dtype=np.int64)
    set_names = []
    lay = -1
    for es in d["sets"]["element"]:
        labels = es.get("labels")
        if labels:
            set_names.append(es["name"]); lay += 1
            cell_domain_ids[np.asarray(labels, dtype=np.int64) - 1] = lay

    # --- material params per domain (in set order), looked up by name ---
    def _flat9(m):
        E_, G_, nu_ = m["E"], m["G"], m["nu"]
        return [float(E_[0]), float(E_[1]), float(E_[2]),
                float(G_[0]), float(G_[1]), float(G_[2]),
                float(nu_[0]), float(nu_[1]), float(nu_[2])]
    material_param = np.array([_flat9(mat_lookup[nm]) for nm in set_names], dtype=np.float64)  # (n_dom, 9)
    n_dom = len(set_names)
    mat_seq = np.arange(n_dom, dtype=np.int64)               # each domain is its own material
    mat_angles = np.zeros(n_dom, dtype=np.float64)           # fibre orientation is entirely in elem_rotation

    # --- per-element orientation (3x3 frame) ---
    # YAML stores each frame [e1,e2,e3] in cross-section axis order (x, y, beam=z). Beam_solid's
    # rotate_C_with_matrix expects [e1,e2,e3] in the homogenizer global order (beam, x, y), so permute
    # each vector's components (x,y,z) -> (z,x,y): index map [2,0,1] applied per 3-vector.
    eo = np.array([[float(v) for v in _row(o)] for o in d["elementOrientations"]], dtype=np.float64)  # (E, 9)
    elem_rotation = eo[:, [2, 0, 1, 5, 3, 4, 8, 6, 7]]

    return dict(
        n_sg=n_sg, elem_type=elem_type, points=points, cells=cells,
        cell_domain_ids=cell_domain_ids, elem_rotation=elem_rotation,
        mat_seq=mat_seq, mat_angles=mat_angles, material_param=material_param,
        num_nodes=points.shape[0], num_elems=E, num_domains=n_dom,
    )


if __name__ == "__main__":
    import sys
    sg = read_solid_yaml(sys.argv[1])
    print("n_sg=%d  elem_type=%s  V=%d  E=%d  domains=%d  mats=%d"
          % (sg["n_sg"], sg["elem_type"], sg["num_nodes"], sg["num_elems"], sg["num_domains"],
             sg["material_param"].shape[0]))
    print("points bbox:", sg["points"].min(0), sg["points"].max(0))
    print("cell_domain_ids unique:", np.unique(sg["cell_domain_ids"]))
    print("material_param[0]:", sg["material_param"][0])
    print("elem_rotation[0]:", sg["elem_rotation"][0])
