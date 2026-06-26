"""Debug st15 EI3 (+13.8%) and C13 (~2x), and check shear locking in the EB
bending cases (reduced vs full integration on the real section)."""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix
from fe_jax.msg_mesh import read_mesh, mesh_curvature
from msg_rm import assemble_rm, solve_eb
from transverse_shear import transverse_shear_stiffness

YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
VK = {"EA": 1.3082688863e10, "GJ": 1.3157769625e8, "EI2": 1.6630291239e9,
      "EI3": 5.1066629243e9, "C13": 1.4345965587e7, "C14": -3.5711027657e9}

n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
nodes, cells, lpe = read_mesh(n3d, elements, e2l)
nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))
D_by = {ln: np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        for ln, i in layup_db.items()}
G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
        for ln, i in layup_db.items()}


def run(reduced):
    ndof = 5*len(nodes2d)
    Kqq = np.zeros((ndof, ndof)); Kqe = np.zeros((ndof, 4)); Kee = np.zeros((4, 4))
    Kee_by = {}
    for e in range(len(elems)):
        kqq, kqe, kee = assemble_rm(nodes2d, elems[e:e+1], ndof, D_by[lpe[e]],
                                    G_by[lpe[e]], k22[e:e+1], p=1, reduced=reduced)
        Kqq += kqq; Kqe += kqe; Kee += kee
        Kee_by[lpe[e]] = Kee_by.get(lpe[e], 0.0) + kee[3, 3]   # EI3 macro per layup
    return solve_eb(Kqq, Kqe, Kee, nodes2d), Kee, Kee_by


print("=== shear-locking check: reduced vs full integration (st15) ===")
Cr, Keer, Kee_by = run(True)
Cf, Keef, _ = run(False)
for i, nm in enumerate(["EA", "GJ", "EI2", "EI3"]):
    print(f"  {nm:4s} reduced {Cr[i,i]:.4e}  full {Cf[i,i]:.4e}  "
          f"diff {100*(Cf[i,i]-Cr[i,i])/Cr[i,i]:+.2f}%")
print(f"  C13  reduced {Cr[0,2]:.4e}  full {Cf[0,2]:.4e}")
print("  -> if reduced==full, no locking in these terms.\n")

print("=== EI3 anatomy (VABS 5.107e9; RM 5.809e9 = +13.8%) ===")
print(f"  macro K_ee[3,3]      = {Keer[3,3]:.4e}")
print(f"  final EI3 (w/ fluc)  = {Cr[3,3]:.4e}  (fluctuation drop "
      f"{100*(Cr[3,3]-Keer[3,3])/Keer[3,3]:+.1f}%)")
print("  EI3 macro contribution by layup (top 6):")
for ln, v in sorted(Kee_by.items(), key=lambda kv: -abs(kv[1]))[:6]:
    h = sum(layup_db[ln]['thick'])
    print(f"    {ln:10s} {v:11.4e}  ({100*v/Keer[3,3]:5.1f}%)  h={h*1e3:.0f}mm")

print("\n=== C13 = EA * tension-center-y3 ===")
xt3 = Cr[0, 2]/Cr[0, 0]; xt3v = VK["C13"]/VK["EA"]
print(f"  RM   xt3 = {xt3*1e3:+.3f} mm   (C13 {Cr[0,2]:.3e})")
print(f"  VABS xt3 = {xt3v*1e3:+.3f} mm  (C13 {VK['C13']:.3e})")
print(f"  -> C13 'error' = EA * {(xt3-xt3v)*1e3:+.3f} mm  (tension-center height,"
      " a reference/thick-cap effect; same as Kirchhoff, not RM/locking).")
