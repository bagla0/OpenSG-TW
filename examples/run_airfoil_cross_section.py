"""
General Cross-Section MSG Shell Beam Homogenization

Reads an OpenSG-format YAML cross-section file and computes Timoshenko
beam stiffness using the MSG Kirchhoff shell model with quadratic Lagrange
elements along the cross-section arc.

Usage
-----
    python examples/run_airfoil_cross_section.py [path/to/cross_section.yaml]

Default YAML (no argument): OpenSG example 1Dshell_0.yaml
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import pypardiso

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONIOENCODING"] = "utf-8"
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=6, linewidth=120)

# All MSG shell functions come from the fe_jax package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
from fe_jax import (
    # materials / ABD
    compute_ABD_matrix,
    compute_ABD_CLT,
    # mesh
    load_yaml,
    order_mesh,
    mesh_curvature,
    # FEM assembly
    gauss_legendre_01,
    compute_element_geometry,
    build_periodic_dof_map,
    compress_dof_map,
    assemble_system_matrices,
    build_lagrange_constraints,
    build_psi_matrix,
    # solvers
    solve_fluctuation_field,
    prepare_v1_rhs,
    finalize_v1_and_compute_deff,
)


def run_cross_section(yaml_path):
    t0 = time.time()
    print("=" * 64)
    print("MSG Shell Beam Homogenization  |  Quadratic Lagrange Elements")
    print("=" * 64)
    print(f"YAML: {yaml_path}\n")

    # ------------------------------------------------------------------ load
    nodes_3d, elements, material_db, layup_db, elem_to_layup = load_yaml(yaml_path)
    print(f"Nodes: {len(nodes_3d)},  Elements: {len(elements)}")
    print(f"Materials : {', '.join(material_db)}")
    print(f"Layups    : {', '.join(layup_db)}")

    # --------------------------------------------------------- ABD per layup
    print("\n--- ABD (MSG Plate Model, quadratic 1D SG) ---")
    ABD_dict  = {}
    mass_dict = {}
    for ln, info in layup_db.items():
        ABD_msg, mass = compute_ABD_matrix(
            info['thick'], info['angles'], info['mat_names'], material_db)
        ABD_clt = compute_ABD_CLT(
            info['thick'], info['angles'], info['mat_names'], material_db)
        ABD_dict[ln]  = ABD_msg
        mass_dict[ln] = mass
        h = sum(info['thick'])
        dA = abs(ABD_msg[0,0] - ABD_clt[0,0]) / abs(ABD_clt[0,0]) * 100
        dD = abs(ABD_msg[3,3] - ABD_clt[3,3]) / abs(ABD_clt[3,3]) * 100
        print(f"  {ln}: {len(info['thick'])} layers, h={h:.4f} m")
        print(f"    MSG  A11={ABD_msg[0,0]:.4e}  D11={ABD_msg[3,3]:.4e}")
        print(f"    CLT  A11={ABD_clt[0,0]:.4e}  D11={ABD_clt[3,3]:.4e}")
        print(f"    diff A11={dA:.4f}%,  D11={dD:.4f}%")

    # ----------------------------------------------------------- order mesh
    print("\n--- Mesh ---")
    nodes_2d, cells, layup_per_elem, is_closed = order_mesh(
        nodes_3d, elements, elem_to_layup)
    n_elem  = cells.shape[0]
    n_nodes = len(nodes_2d)
    print(f"  {n_elem} elements,  {n_nodes} nodes (incl. midside),  closed={is_closed}")

    L_e, xd2, xd3 = compute_element_geometry(nodes_2d, cells)
    k22 = jnp.array(mesh_curvature(nodes_2d, cells, elements, is_closed))
    flat = len(elements[0]) < 3
    print(f"  Arc length: {float(jnp.sum(L_e)):.6f} m")
    print(f"  Elements  : {'flat 2-node (k22=0)' if flat else 'curved 3-node'}")
    print(f"  k22 range : [{float(k22.min()):.4f}, {float(k22.max()):.4f}]")

    ABD_elems = jnp.stack([
        jnp.array(ABD_dict[ln], dtype=jnp.float64) for ln in layup_per_elem])

    # ----------------------------------------------------------- DOF setup
    dof_map, n_unique = build_periodic_dof_map(n_nodes, cells, is_closed)
    red_cells, _, n_primal = compress_dof_map(dof_map, cells)
    print(f"  Unique nodes: {n_unique},  Primal DOFs: {n_primal}")

    xi_q, W_q = gauss_legendre_01(4)

    # ----------------------------------------------------------- assembly
    print("\n--- Assembly ---")
    t1 = time.time()
    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices(
        jnp.array(nodes_2d, dtype=jnp.float64),
        cells, red_cells, ABD_elems, k22,
        L_e, xd2, xd3, xi_q, W_q, n_primal)
    print(f"  Time: {time.time()-t1:.2f} s")

    C_mat = build_lagrange_constraints(
        jnp.array(nodes_2d, dtype=jnp.float64),
        cells, red_cells, L_e, xi_q, W_q, n_primal)
    Psi = build_psi_matrix(
        jnp.array(nodes_2d[:n_unique], dtype=jnp.float64),
        n_unique, n_primal)
    Dc = C_mat.T

    # No interior penalty needed: the derivative-form twist constraint
    # (build_lagrange_constraints) makes the V1 RHS orthogonal to the rigid
    # kernel, so the C0 warping null modes stay harmless.

    # ------------------------------------------ Euler-Bernoulli (EB) solve
    print("\n--- Euler-Bernoulli solve ---")
    t2 = time.time()
    V0, D1_V0, A_aug = solve_fluctuation_field(Dhh, -np.array(Dhe.todense()), C_mat)
    Ceff = Dee + D1_V0
    print(f"  Time: {time.time()-t2:.2f} s")

    lbl4 = ["EA  (gamma11)", "GJ  (kappa1) ", "EI2 (kappa2) ", "EI3 (kappa3) "]
    for i, lb in enumerate(lbl4):
        print(f"    {lb} = {float(Ceff[i,i]):.6e}")

    # ---------------------------------------------- Timoshenko solve
    print("\n--- Timoshenko solve ---")
    t3 = time.time()
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)

    R_v1 = np.concatenate(
        [np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)

    Ceff_srt, _, _, _ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n_primal, :]), V0, Ceff,
        V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    Ceff_srt.block_until_ready()
    print(f"  Time: {time.time()-t3:.2f} s")

    lbl6 = ["EA  ", "GA12", "GA13", "GJ  ", "EI2 ", "EI3 "]
    print("\n  Timoshenko 6x6 stiffness  "
          "[gamma11, gamma12, gamma13, kappa1, kappa2, kappa3]:")
    for i, lb in enumerate(lbl6):
        print(f"    {lb} = {float(Ceff_srt[i,i]):.6e}")

    print("\n  Full 6x6:")
    print(np.array(Ceff_srt))

    mass_per_m = sum(mass_dict[layup_per_elem[i]][0] * float(L_e[i])
                     for i in range(n_elem))
    print(f"\n  Mass per unit beam length: {mass_per_m:.4f} kg/m")
    print(f"\n{'='*64}\nTotal time: {time.time()-t0:.2f} s")
    return Ceff_srt


if __name__ == "__main__":
    default_yaml = os.path.join(
        r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG", "1Dshell_0.yaml")
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else default_yaml
    run_cross_section(yaml_path)
