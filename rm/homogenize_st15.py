"""Station-15 EB 4x4 beam stiffness via the RM (C0) cross-section homogenization,
compared with VABS (.K classical) and the Kirchhoff (Hermite) model.

EB order [gamma11, kappa1(twist), kappa2, kappa3] = VABS classical [ext,twist,bend2,bend3].
"""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, timoshenko_from_yaml, compute_ABD_matrix
from fe_jax.msg_mesh import read_mesh, mesh_curvature
from fe_jax.msg_materials import rotation_6x6
from msg_rm import assemble_rm, solve_eb

YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
# VABS .K classical 4x4 [ext, twist, bend2, bend3]
VK = np.array([[1.3082688863e10, 0, 1.4345965587e7, -3.5711027657e9],
               [0, 1.3157769625e8, 0, 0],
               [1.4345965587e7, 0, 1.6630291239e9, 2.5855045936e8],
               [-3.5711027657e9, 0, 2.5855045936e8, 5.1066629243e9]])
LBL = ["EA  ", "GJ  ", "EI2 ", "EI3 "]


def G_layup(info, mat_db, kcorr=5/6):
    """2x2 transverse-shear stiffness of a layup: sum_k R2(theta) diag(G13,G23) R2^T t."""
    Gs = np.zeros((2, 2))
    for t, ang, mn in zip(info["thick"], info["angles"], info["mat_names"]):
        G13, G23 = mat_db[mn]["G"][1], mat_db[mn]["G"][2]
        th = np.deg2rad(ang); c, s = np.cos(th), np.sin(th)
        R2 = np.array([[c, s], [-s, c]])
        Gs += R2 @ np.diag([G13, G23]) @ R2.T * t
    return kcorr * Gs


def main():
    nodes3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
    nodes, cells, layup_per_elem = read_mesh(nodes3d, elements, e2l)
    k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))

    D_by = {ln: np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"],
            mat_db)[0]) for ln, i in layup_db.items()}
    G_by = {ln: G_layup(i, mat_db) for ln, i in layup_db.items()}
    nodes2d = nodes[:, :2]
    elems = cells[:, [0, 1]]                     # 2-node linear CG

    # per-element D, G via wrapper that indexes by layup
    De = [D_by[ln] for ln in layup_per_elem]
    Ge = [G_by[ln] for ln in layup_per_elem]

    # assemble (assemble_rm takes single D,Gs; extend to per-element)
    Kqq, Kqe, Kee = _assemble_perelem(nodes2d, elems, De, Ge, k22)
    C_rm = solve_eb(Kqq, Kqe, Kee, nodes2d)

    EB_k, _, _ = timoshenko_from_yaml(YAML, frac=0.0)
    C_k = np.asarray(EB_k)

    print("Station 15 EB 4x4  [ext, twist, bend2, bend3]\n")
    print(f"  {'term':6s}{'RM (C0)':>14s}{'Kirchhoff':>14s}{'VABS .K':>14s}"
          f"{'RM%':>8s}{'KF%':>8s}")
    for i, lbl in enumerate(LBL):
        v = VK[i, i]
        print(f"  {lbl:6s}{C_rm[i,i]:14.4e}{C_k[i,i]:14.4e}{v:14.4e}"
              f"{100*(C_rm[i,i]-v)/v:8.1f}{100*(C_k[i,i]-v)/v:8.1f}")
    print("\n  couplings (ext-bend):")
    for (i, j), nm in [((0, 2), "C13 ext-bend2"), ((0, 3), "C14 ext-bend3")]:
        v = VK[i, j]
        print(f"    {nm:14s} RM {C_rm[i,j]:+.4e}  KF {C_k[i,j]:+.4e}  VABS {v:+.4e}")


def _assemble_perelem(nodes, elems, De, Ge, k22):
    """assemble_rm but with per-element D,G (loops one element 'mesh' at a time
    and accumulates -- reuses the validated single-material kernel)."""
    ndof = 5 * len(nodes)
    Kqq = np.zeros((ndof, ndof)); Kqe = np.zeros((ndof, 4)); Kee = np.zeros((4, 4))
    for e in range(len(elems)):
        kqq, kqe, kee = assemble_rm(nodes, elems[e:e+1], ndof, De[e], Ge[e],
                                    k22[e:e+1], p=1, reduced=True)
        Kqq += kqq; Kqe += kqe; Kee += kee
    return Kqq, Kqe, Kee


if __name__ == "__main__":
    main()
