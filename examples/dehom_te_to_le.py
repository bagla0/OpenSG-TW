"""
Example: MSG thin-walled (TW) dehomogenization for a PRESCRIBED beam strain.

Given a user-supplied global beam (generalized) strain, recover the local 3D
stress on the airfoil outer mold line (OML) and report it along the upper-surface
trailing-edge -> leading-edge path.

Two-step recovery (see opensg_jax/fe_jax/msg_dehom.py):
  step 1  shell strains : recover_shell_strains(bundle, beam_strain=..., xi_eval)
  step 2  plate dehom   : plate_dehom_strain  -> 3D stress through the thickness;
                          the OML value is the z=0 (reference-surface) point.

The recovered laminate-frame stress is rotated to the global/beam frame
(y1=beam, y2, y3) so the components are physical [S11,S22,S33,S23,S13,S12].

Usage
-----
    python examples/dehom_te_to_le.py [path/to/1Dshell.yaml]

Edit BEAM_STRAIN below for your load case.
"""
import os
import sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import (solve_tw_from_yaml, recover_shell_strains,
                    compute_ABD_matrix, plate_dehom_strain)

# ============================ USER INPUT =====================================
# Global beam strain in VABS order:
#   [eps11 (axial), gamma12, gamma13 (transverse shears),
#    kappa1 (twist), kappa2, kappa3 (bending)]
BEAM_STRAIN = [1.0e-3, 0.0, 0.0, 0.0, 5.0e-4, 0.0]

DEFAULT_YAML = os.path.join(
    r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG", "1Dshell_0.yaml")
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
# =============================================================================


def _voigt_to_tensor(s):
    return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


def _tensor_to_voigt(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[1, 2], T[0, 2], T[0, 1]])


def _upper_te_to_le(xy):
    """Indices of upper-surface OML nodes ordered trailing-edge -> leading-edge."""
    le = int(np.argmin(xy[:, 0])); te = int(np.argmax(xy[:, 0]))
    chord = xy[te] - xy[le]
    cross = chord[0] * (xy[:, 1] - xy[le, 1]) - chord[1] * (xy[:, 0] - xy[le, 0])
    side = cross >= 0
    if xy[side, 1].mean() < xy[~side, 1].mean():
        side = ~side
    up = np.where(side)[0]
    return up[np.argsort(-xy[up, 0])]


def oml_stress(yaml_path, beam_strain):
    """Per-node OML 3D stress (global frame) for a prescribed beam strain."""
    bundle = solve_tw_from_yaml(yaml_path)

    # step 1: shell strains at the element end nodes (xi=0,1) -> nodal values
    sh = recover_shell_strains(bundle, beam_strain=beam_strain, xi_eval=[0.0, 1.0])
    rc = np.asarray(bundle["red_cells"]); corners = np.asarray(bundle["corners"])
    tang = sh["tang"]; ss = sh["shell_strain"]; layups = bundle["layup_per_elem"]
    nn = corners.shape[0]
    nod_ss = np.zeros((nn, 6)); nod_t = np.zeros((nn, 2)); cnt = np.zeros(nn)
    nod_layup = [None] * nn
    for e in range(rc.shape[0]):
        for loc in (0, 1):
            nd = int(rc[e, loc]); nod_ss[nd] += ss[e, loc]
            nod_t[nd] += tang[e]; cnt[nd] += 1; nod_layup[nd] = layups[e]
    nod_ss /= cnt[:, None]; nod_t /= np.linalg.norm(nod_t, axis=1, keepdims=True)

    # step 2: plate dehom per node -> OML (z=0) stress, rotate laminate -> global
    warp = {ln: compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"],
            bundle["material_db"], return_warping=True)[2]
            for ln, i in bundle["layup_db"].items()}
    glob = np.zeros((nn, 6))
    for nd in range(nn):
        z, _, Sig = plate_dehom_strain(warp[nod_layup[nd]], nod_ss[nd], 3)
        s_lam = Sig[int(np.argmin(z))]                  # z=0 = OML
        t2, t3 = nod_t[nd]; n2, n3 = t3, -t2
        R = np.array([[1., 0, 0], [0, t2, n2], [0, t3, n3]])
        glob[nd] = _tensor_to_voigt(R @ _voigt_to_tensor(s_lam) @ R.T)
    return corners, glob


def main(yaml_path):
    st = np.asarray(BEAM_STRAIN, float)
    print("=" * 70)
    print("MSG-TW dehomogenization  |  prescribed beam strain")
    print("=" * 70)
    print(f"YAML        : {yaml_path}")
    print("beam strain : [eps11, gamma12, gamma13, kappa1, kappa2, kappa3]")
    print(f"            = {st}\n")

    corners, glob = oml_stress(yaml_path, st)
    idx = _upper_te_to_le(corners)
    xy = corners[idx]; S = glob[idx]
    seg = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    s = seg / seg[-1]

    print(f"Upper-surface OML, TE -> LE ({len(idx)} nodes):")
    print("   s     y2      y3   " + " ".join(f"{c:>10s}" for c in COMP) + "   (Pa)")
    for i in range(len(idx)):
        print(f"  {s[i]:.2f} {xy[i,0]:7.3f} {xy[i,1]:6.3f} " +
              " ".join(f"{S[i,j]:10.3e}" for j in range(6)))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("MSG-TW OML stress (upper surface, TE->LE)  |  beam strain ["
                     + ", ".join(f"{v:g}" for v in st) + "]",
                     fontsize=12, fontweight="bold")
        for j, c in enumerate(COMP):
            ax = axes.flat[j]
            ax.plot(s, S[:, j] / 1e6, "b-o", ms=4)
            ax.set_title(f"$\\sigma_{{{c[1:]}}}$  ({c})", fontweight="bold")
            ax.set_xlabel("path s  (0 = TE  ->  1 = LE)")
            ax.set_ylabel(f"{c}  (MPa)"); ax.grid(True, ls=":", alpha=0.7)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(os.path.dirname(__file__), "..", "outputs")
        os.makedirs(out, exist_ok=True)
        p = os.path.join(out, "dehom_te_to_le.png")
        fig.savefig(p, dpi=150)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_YAML)
