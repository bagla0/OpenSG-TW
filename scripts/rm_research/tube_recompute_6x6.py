"""Recompute the anisotropic -45 tube RM/KF 6x6 (centre reference, h/R=0.121)
with the current code, to refresh the paper's Table tab:homo C44 (GJ) after the
twist-measure fix. Order [ext, shear2, shear3, twist, bend2, bend3]."""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import (read_mesh, mesh_curvature, offset_oml_to_iml,
                             element_e3_from_yaml)
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

YAML = os.path.join(HERE, "..", "outputs", "tube_dehom", "aniso_tube.yaml")
LBL = ["ext", "shear2", "shear3", "twist", "bend2", "bend3"]


def shell_6x6(frac=0.5):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(YAML)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))

    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(a, frac*float(sum(i["thick"]))) if frac else a
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    _, KF, _ = timoshenko_from_yaml(YAML, frac=frac)
    return np.asarray(RM), np.asarray(KF)


RM, KF = shell_6x6(0.5)
print("Anisotropic -45 tube, centre reference (h/R=0.121), current code:")
for (i, j) in [(0, 0), (0, 3), (1, 1), (2, 2), (1, 4), (3, 3), (4, 4), (5, 5)]:
    print(f"  C{i+1}{j+1} ({LBL[i][:3]}-{LBL[j][:3]}):  RM={RM[i,j]:.4e}   KF={KF[i,j]:.4e}")
