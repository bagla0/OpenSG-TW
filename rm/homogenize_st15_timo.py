"""Station-15 full 6x6 Timoshenko via the RM (C0) model + V1, vs VABS .K and
the Kirchhoff model.  RM 6x6 order [EA, GA12, GA13, GJ, EI2, EI3] = VABS
Timoshenko [ext, shear2, shear3, twist, bend2, bend3]."""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import read_mesh, mesh_curvature
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
# VABS .K Timoshenko 6x6 [ext, shear2, shear3, twist, bend2, bend3]
VK = np.array([
 [1.3082688863e10, 0, 0, 0, 1.4345965587e7, -3.5711027657e9],
 [0, 4.5798883250e8, -2.3551934467e7, -2.1795975869e7, 0, 0],
 [0, -2.3551934467e7, 1.0549929775e8, 5.0551071833e7, 0, 0],
 [0, -2.1795975869e7, 5.0551071833e7, 1.5604378583e8, 0, 0],
 [1.4345965587e7, 0, 0, 0, 1.6630291239e9, 2.5855045936e8],
 [-3.5711027657e9, 0, 0, 0, 2.5855045936e8, 5.1066629243e9]])
LBL = ["EA  ", "GA12", "GA13", "GJ  ", "EI2 ", "EI3 "]


def main():
    n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))
    D_by = {ln: np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
            for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}

    C6, EB = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    _, Ck, _ = timoshenko_from_yaml(YAML, frac=0.0); Ck = np.asarray(Ck)

    print("Station 15 Timoshenko 6x6  [EA, GA12, GA13, GJ, EI2, EI3]\n")
    print(f"  {'term':6s}{'RM V1':>14s}{'Kirchhoff':>14s}{'VABS .K':>14s}{'RM%':>8s}{'KF%':>8s}")
    for i, lbl in enumerate(LBL):
        v = VK[i, i]
        print(f"  {lbl:6s}{C6[i,i]:14.4e}{Ck[i,i]:14.4e}{v:14.4e}"
              f"{100*(C6[i,i]-v)/v:8.1f}{100*(Ck[i,i]-v)/v:8.1f}")
    print("\n  transverse-shear focus (the V1 payoff):")
    for i, lbl in [(1, "GA12 (C22)"), (2, "GA13 (C33)")]:
        print(f"    {lbl:12s} RM {C6[i,i]:.4e}  KF {Ck[i,i]:.4e}  VABS {VK[i,i]:.4e}"
              f"   RM {100*(C6[i,i]-VK[i,i])/VK[i,i]:+.1f}%  KF {100*(Ck[i,i]-VK[i,i])/VK[i,i]:+.1f}%")


if __name__ == "__main__":
    main()
