"""
General Cross-Section MSG Shell Beam Homogenization

Reads an OpenSG-format YAML cross-section file and computes Timoshenko
beam stiffness using the MSG Kirchhoff shell model with Hermite C1 cubic
shape functions for the displacement along the cross-section arc (value +
arc-slope DOFs at the corner nodes).  The line cross-section geometry uses
the quadratic 3-node element, so curvature k22 is captured (flat 2-node
elements give k22=0).  Frame: VABS plane cross-section (e1=[1,0,0]).  No
interior penalty is needed -- C1 continuity removes the spurious C0 modes.

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
    # mesh (direct from YAML connectivity — no chaining)
    load_yaml,
    read_mesh,
    mesh_curvature,
    # FEM assembly (Hermite C1)
    gauss_legendre_01,
    compute_element_geometry,
    assemble_system_matrices_hermite,
    build_constraints_hermite,
    # solvers
    solve_fluctuation_field,
    prepare_v1_rhs,
    finalize_v1_and_compute_deff,
)


def run_cross_section(yaml_path):
    t0 = time.time()
    print("=" * 64)
    print("MSG Shell Beam Homogenization  |  Hermite C1 Cubic Elements")
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

    # ------------------------------- mesh: YAML connectivity verbatim (no chain)
    print("\n--- Mesh (direct from YAML connectivity) ---")
    nodes, cells, layup_per_elem = read_mesh(nodes_3d, elements, elem_to_layup)
    n_elem = cells.shape[0]
    flat = cells.shape[1] < 3
    k22 = jnp.array(mesh_curvature(nodes, cells, elements, is_closed=False))
    print(f"  {n_elem} line elements,  {len(nodes)} nodes  "
          f"({'flat 2-node, k22=0' if flat else 'curved 3-node'})")
    print(f"  k22 range : [{float(k22.min()):.4f}, {float(k22.max()):.4f}]")

    ABD_elems = jnp.stack([
        jnp.array(ABD_dict[ln], dtype=jnp.float64) for ln in layup_per_elem])

    # ------------------------------------------- Hermite C1 DOF setup
    # Displacement = Hermite C1 cubic (value+slope at corner nodes, 6 DOF/node).
    hcells = cells[:, [0, -1]]
    used = np.unique(hcells)
    f2r = np.full(nodes.shape[0], -1, dtype=np.int64)
    f2r[used] = np.arange(len(used))
    red_cells = f2r[hcells]
    corners = nodes[used]
    n_unique = len(used)
    n_primal = 6 * n_unique
    Lh, xd2h, xd3h = compute_element_geometry(corners, red_cells)
    print(f"  Nodes used: {n_unique},  Primal DOFs (Hermite C1): {n_primal}")

    xi_q, W_q = gauss_legendre_01(4)

    # ----------------------------------------------------------- assembly
    print("\n--- Assembly (Hermite C1) ---")
    t1 = time.time()
    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices_hermite(
        corners, red_cells, red_cells, ABD_elems, k22,
        Lh, xd2h, xd3h, xi_q, W_q, n_primal)
    print(f"  Time: {time.time()-t1:.2f} s")

    C_mat, Psi = build_constraints_hermite(
        corners, red_cells, red_cells, Lh, xd2h, xd3h, xi_q, W_q, n_primal, n_unique)
    Dc = C_mat.T

    # No interior penalty needed: Hermite C1 has continuous slope (no spurious
    # C0 warping modes), and the derivative-form twist constraint keeps the V1
    # RHS orthogonal to the rigid kernel.

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

    mass_per_m = sum(mass_dict[layup_per_elem[i]][0] * float(Lh[i])
                     for i in range(n_elem))
    print(f"\n  Mass per unit beam length: {mass_per_m:.4f} kg/m")
    print(f"\n{'='*64}\nTotal time: {time.time()-t0:.2f} s")
    return Ceff_srt


if __name__ == "__main__":
    default_yaml = os.path.join(
        r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG", "1Dshell_0.yaml")
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else default_yaml
    run_cross_section(yaml_path)
