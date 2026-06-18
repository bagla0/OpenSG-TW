"""
Reissner-Mindlin (RM) thin-walled MSG shell -> Timoshenko 6x6 stiffness.

Reads the 1D structure-genome strip_iso_1D.yaml and runs the full MSG-TW solve
(no black-box wrapper): per-ply plate ABD + transverse-shear G -> assemble the
fluctuation stiffness -> KKT solve for the classical warping V0 -> shear-warping
V1 -> condense to the 6x6 Timoshenko stiffness.  Prints the 6x6 at the centre
and the OML references.

RM wall kinematics: C0 Lagrange, 5 DOF/node [w1,w2,w3,omega1,omega2]; transverse
shear retained (selective reduced integration); drilling omega3 eliminated by
eps12=eps21.  Order [ext, shear2, shear3, twist, bend2, bend3].

Run (Windows):
  $env:PATH = "C:\\conda_envs\\opensg_2_0_env;...;" + $env:PATH   # see CLAUDE.md
  & "C:\\conda_envs\\opensg_2_0_env\\python.exe" strip_RM.py
"""
import os, sys
import numpy as np
from scipy.sparse import coo_matrix
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "rm"))
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pypardiso

from fe_jax import load_yaml, compute_ABD_matrix
from fe_jax.msg_mesh import read_mesh, offset_oml_to_iml, element_e3_from_yaml
from fe_jax.msg_materials import shift_abd_reference
from fe_jax.msg_solver import (solve_fluctuation_field, prepare_v1_rhs,
                               finalize_v1_and_compute_deff)
from msg_rm_timo import assemble_all, build_C_Psi          # RM element operators
from transverse_shear import transverse_shear_stiffness    # MSG plate G block

YAML = os.path.join(HERE, "strip_iso_1D.yaml")
LBL = ["ext", "shear2", "shear3", "twist", "bend2", "bend3"]


def rm_timoshenko_6x6(yaml_path, frac):
    """Full RM MSG-TW solve for the Timoshenko 6x6 at reference `frac`
    (0.0 = OML, 0.5 = wall mid-surface / centre)."""
    # 1. read the 1D genome and build the line mesh
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:                                                # offset OML -> reference
        e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.zeros(len(elems))                              # flat strip -> no curvature

    # 2. per-ply plate stiffness: 6x6 ABD (1D through-thickness SG) + 2x2 shear G
    def D_of(i):
        abd = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(abd, frac*float(sum(i["thick"]))) if frac else abd
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}

    # 3. assemble the RM fluctuation system (Dhh warping, Dhe/Dee macro,
    #    Dhl/Dll/Dle the second-order shear-warping operators)
    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_all(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    C, Psi = build_C_Psi(nodes2d, elems, p=1)               # 4 constraints + rigid kernel
    Dc = C.T

    # 4. KKT solve for the classical (Euler-Bernoulli) warping V0 -> 4x4 D_eff
    V0, D1, A_aug = solve_fluctuation_field(coo_matrix(Dhh), -Dhe, Dc)
    Deff = Dee + np.asarray(D1)

    # 5. shear-warping (V1) solve and condensation to the 6x6 Timoshenko stiffness
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle),
        jnp.array(Psi), jnp.array(Dc))
    n = Dhh.shape[0]
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n, :]), jnp.array(V0), jnp.array(Deff),
        V0DllV0, DhlV0, DhlTV0Dle, jnp.array(Psi), jnp.array(Dc))
    return np.asarray(C6)


def show(tag, C6):
    print(f"\n=== RM Timoshenko 6x6  ({tag})  order {LBL} ===")
    for i in range(6):
        print("  " + "".join(f"{C6[i, j]:14.4e}" for j in range(6)))
    d = np.diag(C6)
    print("  diagonal:  EA={:.4e}  GA2={:.4e}  GA3={:.4e}  GJ={:.4e}  EI2={:.4e}  EI3={:.4e}"
          .format(d[0], d[1], d[2], d[3], d[4], d[5]))


if __name__ == "__main__":
    print(f"reading 1D genome: {YAML}")
    show("centre / mid-surface", rm_timoshenko_6x6(YAML, frac=0.5))
    show("OML", rm_timoshenko_6x6(YAML, frac=0.0))
