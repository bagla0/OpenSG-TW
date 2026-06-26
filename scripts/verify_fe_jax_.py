"""Smoke-test the consolidated fe_jax_ package: import it and compute the strip
Kirchhoff and RM Timoshenko 6x6 (centre), confirming it is self-contained."""
import os, sys
import numpy as np
from scipy.sparse import coo_matrix
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pypardiso
import fe_jax_ as fj

YAML = os.path.join(HERE, "strip_iso_1D.yaml")

# --- Kirchhoff (one call) ---
_, KF, _ = fj.timoshenko_from_yaml(YAML, frac=0.5)
KF = np.asarray(KF)

# --- RM (assemble -> KKT V0 -> V1 -> 6x6, all from fe_jax_) ---
n3d, elements, mat_db, layup_db, e2l = fj.load_yaml(YAML)
nodes, cells, lpe = fj.read_mesh(n3d, elements, e2l)
e3 = fj.element_e3_from_yaml(YAML)
nodes = fj.offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=0.5)
nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]; k22 = np.zeros(len(elems))
D_by = {ln: fj.shift_abd_reference(
            np.asarray(fj.compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0]),
            0.5*float(sum(i["thick"]))) for ln, i in layup_db.items()}
G_by = {ln: fj.transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
        for ln, i in layup_db.items()}
RM, _ = fj.timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
RM = np.asarray(RM)

print("fe_jax_ imported OK.  KAPPA12_MACRO =", fj.KAPPA12_MACRO)
print("Kirchhoff (centre): EA={:.4e} GA2={:.4e} GJ={:.4e} EI2={:.4e} EI3={:.4e}".format(
    KF[0, 0], KF[1, 1], KF[3, 3], KF[4, 4], KF[5, 5]))
print("RM        (centre): EA={:.4e} GA2={:.4e} GJ={:.4e} EI2={:.4e} EI3={:.4e}".format(
    RM[0, 0], RM[1, 1], RM[3, 3], RM[4, 4], RM[5, 5]))
