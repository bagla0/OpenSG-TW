"""KL homogenization with the reference selected PURELY by the ABD computation.

Per the corrected design: the reference surface (OML/center/IML) is NOT a node
shift and NOT a parallel-axis transform of the assembled ABD -- it is the
mesh-based reference of the through-thickness plate SG, applied via
``compute_ABD_matrix(z_ref=frac*h)`` (the route the docstring marks "preferred").
The mesh YAML carries ONLY the geometry (fixed nodes); a SEPARATE reference YAML
(e.g. inputs/reference_center.yaml) defines ``frac``.

k22 (hoop curvature) is taken from the fixed node geometry, sign tied to the
element traversal -- exactly as in tube_lib, but with no R_ref / d_shift argument.

Timoshenko order: [EA, GA2, GA3, GJ, EI2, EI3].
"""
import os
import sys
import numpy as np
import yaml as _yaml

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
LIB = os.path.join(CC, "examples", "TW-paper", "lib")
sys.path.insert(0, LIB)
sys.path.insert(0, os.path.join(CC, "opensg_jax"))

import jax
jax.config.update("jax_enable_x64", True)

from fe_jax import load_yaml, compute_ABD_matrix        # noqa: E402
from fe_jax.msg_mesh import read_mesh                   # noqa: E402
from tube_lib import _kirchhoff, _flipB                 # noqa: E402


def load_reference(ref_yaml):
    with open(ref_yaml) as f:
        d = _yaml.safe_load(f)
    return float(d.get("frac", 0.0)), d.get("reference", "OML")


def homog_kl(mesh_yaml, ref_yaml, e3="inward"):
    """KL Timoshenko 6x6 for ``mesh_yaml`` with the reference from ``ref_yaml``.
    Reference is applied ONLY through compute_ABD_matrix(z_ref=frac*h); nodes are
    used verbatim from the mesh (no shift)."""
    frac, _name = load_reference(ref_yaml)
    n3d, elements, mat_db, layup_db, e2l = load_yaml(mesh_yaml)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    nodes2d = nodes[:, :2]
    elems = cells[:, [0, 1]]
    ne = len(elems)

    # hoop curvature from the fixed node geometry: k22 = ksign/|midpoint| per
    # element (exact -1/R for a circle), sign from the traversal (signed area).
    xy = nodes2d[elems[:, 0]]
    area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                             - np.roll(xy[:, 0], -1) * xy[:, 1]))
    ksign = -1.0 if area > 0 else 1.0
    mids = 0.5 * (nodes2d[elems[:, 0]] + nodes2d[elems[:, 1]])
    Rmid = np.linalg.norm(mids, axis=1)
    k22 = ksign / Rmid

    def D_of(i):
        h = float(np.sum(i["thick"]))
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"],
                                          mat_db, z_ref=frac * h)[0])
        return _flipB(a) if e3 == "outward" else a

    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    KF = _kirchhoff(nodes2d, elems, lpe, D_by, k22)
    return np.asarray(KF)


if __name__ == "__main__":
    # quick self-test on the iso + aniso tube meshes (nodes already at mid-wall)
    inp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inputs")
    refc = os.path.join(inp, "reference_center.yaml")
    for tag, mesh, e3 in [("ISO  R5", "iso_tube_R5_h02_n64.yaml", "inward"),
                          ("ANISO -45", "aniso_tube_m45_n92.yaml", "outward")]:
        KF = homog_kl(os.path.join(inp, mesh), refc, e3=e3)
        d = np.diag(KF)
        print(f"\n{tag}  (z_ref=center, no node/parallel-axis shift)")
        print(f"  EA={d[0]/1e6:.4f}  C14={KF[0,3]/1e6:.5f}  GA2={d[1]/1e6:.4f}"
              f"  GJ={d[3]/1e6:.5f}  EI2={d[4]/1e6:.5f}  EI3={d[5]/1e6:.5f}")
