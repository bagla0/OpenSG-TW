"""
MSG-TW (JAX) OML 3D-stress benchmark — step toward shell-vs-solid comparison.

Solves 1Dshell_0 (Hermite C1 Timoshenko), applies a COMMON beam force FF
(defined at the top level, not inside the recovery), dehomogenizes, and writes
the 3D stress at the OML nodes (the 1Dshell reference nodes lie on the OML) in
the GLOBAL/beam frame (y1,y2,y3), ordered leading-edge to trailing-edge.

Output: outputs/oml_jax.txt  with columns
    y2  y3   S11 S22 S33 S23 S13 S12   (global frame, Pa)

The matching FEniCS-solid run (2Dsolid_0, same FF) and the comparison table are
produced by ``benchmark_oml_compare.py``.
"""
import os, sys
import numpy as np
import jax

os.environ["CUDA_VISIBLE_DEVICES"] = ""
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
from fe_jax import (solve_tw_from_yaml, recover_shell_strains,
                    compute_ABD_matrix, plate_dehom_strain)

YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_0.yaml"

# ---- COMMON beam force (VABS order [F1,F2,F3,M1,M2,M3]) — shared with solid ----
FF = np.array([1.0e5, 5.0e4, 5.0e4, 5.0e4, 1.0e5, 1.0e5])


def voigt_to_tensor(s):
    return np.array([[s[0], s[5], s[4]],
                     [s[5], s[1], s[3]],
                     [s[4], s[3], s[2]]])


def tensor_to_voigt(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[1, 2], T[0, 2], T[0, 1]])


def main():
    bundle = solve_tw_from_yaml(YAML)
    print("Timoshenko 6x6 diag:", np.diag(bundle["Timo"]))

    # Shell strains at the element END NODES (xi=0,1) -> nodal values on the OML
    sh = recover_shell_strains(bundle, beam_force_vabs=FF, xi_eval=[0.0, 1.0])
    print("beam macro strain st:", sh["macro"])

    rc = np.asarray(bundle["red_cells"])
    corners = np.asarray(bundle["corners"])
    tang = sh["tang"]                       # (E,2) unit arc tangent per element
    ss = sh["shell_strain"]                 # (E,2,6) at the two end nodes
    layups = bundle["layup_per_elem"]
    n_node = corners.shape[0]

    # accumulate nodal shell strain + nodal tangent (average of adjacent elems)
    nod_ss = np.zeros((n_node, 6)); nod_t = np.zeros((n_node, 2)); cnt = np.zeros(n_node)
    nod_layup = [None] * n_node
    for e in range(rc.shape[0]):
        for loc in (0, 1):
            nd = int(rc[e, loc])
            nod_ss[nd] += ss[e, loc]; nod_t[nd] += tang[e]; cnt[nd] += 1
            nod_layup[nd] = layups[e]
    nod_ss /= cnt[:, None]
    nod_t /= np.linalg.norm(nod_t, axis=1, keepdims=True)

    # plate warping cache per layup
    warp_cache = {}
    for ln, info in bundle["layup_db"].items():
        _, _, warp = compute_ABD_matrix(info["thick"], info["angles"],
            info["mat_names"], bundle["material_db"], return_warping=True)
        warp_cache[ln] = warp

    # OML stress per node, rotated laminate(beam,tangent,normal) -> global(y1,y2,y3)
    out = np.zeros((n_node, 8))
    for nd in range(n_node):
        z, Gam, Sig = plate_dehom_strain(warp_cache[nod_layup[nd]], nod_ss[nd], 3)
        i_oml = int(np.argmin(z))           # z=0 face = OML (reference surface)
        s_lam = Sig[i_oml]
        t2, t3 = nod_t[nd]; n2, n3 = t3, -t2   # normal = tangent rotated -90 deg
        R = np.array([[1.0, 0.0, 0.0], [0.0, t2, n2], [0.0, t3, n3]])
        s_glob = tensor_to_voigt(R @ voigt_to_tensor(s_lam) @ R.T)
        out[nd, :2] = corners[nd]           # (y2, y3)
        out[nd, 2:] = s_glob

    # order leading-edge (min y2) -> trailing-edge (max y2) around the loop:
    # split into upper/lower by sign of y3, sort each by y2, concatenate.
    y2, y3 = out[:, 0], out[:, 1]
    le = np.argmin(y2)
    upper = np.where(y3 >= y3[le])[0]; lower = np.where(y3 < y3[le])[0]
    order = np.concatenate([upper[np.argsort(y2[upper])],
                            lower[np.argsort(-y2[lower])]])
    out = out[order]

    os.makedirs("outputs", exist_ok=True)
    hdr = ("FF=" + ",".join(f"{v:g}" for v in FF) + "\n"
           "y2 y3 S11 S22 S33 S23 S13 S12 (global frame, Pa)")
    np.savetxt("outputs/oml_jax.txt", out, header=hdr, fmt="%18.8e")
    print(f"\nwrote outputs/oml_jax.txt  ({n_node} OML nodes)")
    print("first 6 rows (y2,y3, S11 S22 S33 S23 S13 S12):")
    for r in out[:6]:
        print("  " + " ".join(f"{v: .3e}" for v in r))


if __name__ == "__main__":
    main()
