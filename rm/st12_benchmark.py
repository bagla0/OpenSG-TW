"""
Station-12 stiffness benchmark: thin-walled shell (RM, Kirchhoff) from
1Dshell_12.yaml vs the FEniCS 2D-solid 6x6 from 2Dsolid_12.yaml.
Shell evaluated at both OML (frac=0) and centroid (frac=0.5) references.
Order [ext, shear2, shear3, twist, bend2, bend3].
"""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import (read_mesh, mesh_curvature, offset_oml_to_iml,
                             element_e3_from_yaml)
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

DATA = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
        r"\training data\opensg-FEniCS\data")
YAML = os.path.join(DATA, "1Dshell_12.yaml")
SOLID = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
         r"\outputs\st12\solid_C6_st12.txt")
LBL = ["ext", "shear2", "shear3", "twist", "bend2", "bend3"]


def shell_6x6(frac):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
    n3d = n3d.copy(); n3d[:, 2] = 0.0                      # zero the beam-x column
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(YAML)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))

    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(a, frac*float(sum(i["thick"]))) if frac else a
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    _, KF, _ = timoshenko_from_yaml(YAML, frac=frac)
    return np.asarray(RM), np.asarray(KF)


def fmt(M):
    return "\n".join("  " + "".join(f"{M[i,j]:13.4e}" for j in range(6)) for i in range(6))


def compare(tag, RM, KF, S):
    print(f"\n=== {tag}: shell vs FEniCS-solid (only |solid|>1e6) ===")
    print(f"  {'i,j':7s}{'term':12s}{'solid':>13s}{'RM':>13s}{'KF':>13s}{'RM%':>8s}{'KF%':>8s}")
    sr = sk = sv = 0.0
    for i in range(6):
        for j in range(i, 6):
            v = S[i, j]
            if abs(v) < 1e6:
                continue
            nm = f"{LBL[i][:3]}-{LBL[j][:3]}"
            print(f"  {i+1},{j+1:<4d}{nm:12s}{v:13.4e}{RM[i,j]:13.4e}{KF[i,j]:13.4e}"
                  f"{100*(RM[i,j]-v)/v:8.1f}{100*(KF[i,j]-v)/v:8.1f}")
            sr += (RM[i, j]-v)**2; sk += (KF[i, j]-v)**2; sv += v**2
    print(f"  relative norm:  RM {100*(sr/sv)**0.5:.2f}%   Kirchhoff {100*(sk/sv)**0.5:.2f}%")


def main():
    S = np.loadtxt(SOLID)
    print("Order [ext, shear2, shear3, twist, bend2, bend3]\n")
    print("=== FEniCS 2D-solid st12 Timoshenko 6x6 ===\n" + fmt(S))
    for tag, frac in [("OML (frac=0)", 0.0), ("centroid (frac=0.5)", 0.5)]:
        RM, KF = shell_6x6(frac)
        if frac == 0.5:
            print("\n=== TW-RM (centroid) 6x6 ===\n" + fmt(RM))
            print("\n=== TW-Kirchhoff (centroid) 6x6 ===\n" + fmt(KF))
        compare(tag, RM, KF, S)


if __name__ == "__main__":
    main()
