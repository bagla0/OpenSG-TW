"""Debug st15 GJ (RM, OML reference): isolate the kappa12 macro coefficient and
the transverse-shear contribution.  VABS .K classical GJ = 1.3157769625e8."""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import read_mesh, mesh_curvature
import msg_rm
from msg_rm import assemble_rm, solve_eb
from transverse_shear import transverse_shear_stiffness

YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
VABS_GJ = 1.3157769625e8

n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
nodes, cells, lpe = read_mesh(n3d, elements, e2l)
nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))
D_by = {ln: np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        for ln, i in layup_db.items()}
G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
        for ln, i in layup_db.items()}


def gj(k12, gscale):
    msg_rm.KAPPA12_MACRO = k12
    ndof = 5*len(nodes2d)
    Kqq = np.zeros((ndof, ndof)); Kqe = np.zeros((ndof, 4)); Kee = np.zeros((4, 4))
    for e in range(len(elems)):
        kqq, kqe, kee = assemble_rm(nodes2d, elems[e:e+1], ndof, D_by[lpe[e]],
                                    G_by[lpe[e]]*gscale, k22[e:e+1], p=1, reduced=True)
        Kqq += kqq; Kqe += kqe; Kee += kee
    return solve_eb(Kqq, Kqe, Kee, nodes2d)[1, 1]


print(f"VABS .K classical GJ = {VABS_GJ:.4e}")
print(f"Kirchhoff GJ         = {np.asarray(timoshenko_from_yaml(YAML, frac=0.0)[0])[1,1]:.4e}\n")
print(f"  {'kappa12 macro':16s}{'G on (FSDT)':>16s}{'G off':>16s}")
for k12 in (-2.0, -1.0):
    row = []
    for gs in (1.0, 0.0):
        v = gj(k12, gs); row.append(f"{v:.4e} ({100*(v-VABS_GJ)/VABS_GJ:+.1f}%)")
    print(f"  {k12:<16.1f}{row[0]:>16s}{row[1]:>16s}")
msg_rm.KAPPA12_MACRO = -2.0
